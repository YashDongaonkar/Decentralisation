"""
Decentralised multi-robot warehouse simulation  (pygame)
=========================================================

* Central server  : ONLY broadcasts "pick up item #N at (x, y)".  It never assigns work.
* Robots          : Contract Net Protocol (CNP) among themselves
                      1. server announces a task            (/task_announce)
                      2. every free robot bids its A* cost  (/bids)
                      3. every bidder sees ALL bids and independently computes the
                         same winner (lowest cost, ties -> lowest robot id)
                      4. the winner announces itself        (/winner)
                      5. winner drives to the item (A*), picks it, reports to server
                         (/item_picked), drives to the dropoff, drops it, reports
                         (/item_dropped), then parks out of the way.
* Collision avoid : robots publish their cell + intended next cell every tick
                    (/robot_intent). A robot only moves if nobody is in the target
                    cell and no higher-priority robot wants it. Stuck robots
                    replan around others, or send a /yield_request to the blocker.

Everything between components goes through the MessageBus (publish / subscribe on
named topics), mirroring ROS2 topics, so each class maps 1:1 to a ROS2 node later:
    CentralServer -> server_node      Robot -> robot_node (one per robot, namespaced)
    MessageBus.publish/subscribe -> create_publisher / create_subscription
    Message dataclasses -> .msg files          Clock.now -> node.get_clock().now()
    Warehouse (world state) -> the real world / simulator (Gazebo etc.)

Layout legend (edit LAYOUT below - any size, any number of robots/items/dropoffs):
    '.' floor      '#' empty shelf / obstacle      'B' shelf holding a box (red)
    'R' robot start (purple)                       'D' dropoff (green)

Controls:  SPACE pause | UP/DOWN speed | P toggle A* paths | R restart | ESC quit
"""

import heapq
import itertools
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import pygame

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Reproduces the pixel-art sketch.  (x = column, y = row, origin top-left)
LAYOUT = [
    ".........R",
    ".#.#.#.#..",
    ".#R#.B.#..",
    ".B.#.B.#..",
    ".#.#.#.#..",
    "R#.B.#.#..",
    ".B.#.#.#..",
    ".#.#R#.#..",
    ".#.#.#.B..",
    ".........D",
]

CELL = 60                 # pixel size of one grid cell
MARGIN = 20
TOP_PANEL_H = 150         # height of the central-server text box
SIDE_PANEL_W = 300

# --- timing (seconds of simulated time) ---
STEP_INTERVAL = 0.45      # one grid move per tick
BID_WINDOW = 1.2          # how long robots collect bids before deciding
TASK_GAP = 1.0            # server pause between a winner and the next task
RETRY_DELAY = 2.0         # server pause after a task nobody bid on
CFP_TIMEOUT = BID_WINDOW + 0.8
BUBBLE_TTL = 2.6          # lifetime of robot text boxes

# --- collision avoidance ---
REPLAN_AFTER = 2          # blocked ticks before trying a detour
DETOUR_LIMIT = 10         # max extra steps accepted for a detour (else ask blocker to yield)
HOLD_TICKS = 3            # ticks a robot waits after stepping aside
PARK_MIN_DIST = 3         # parking spots must be this far from any dropoff

# --- colours ---
C_BG = (45,45,45)
C_FLOOR_A = (255, 255, 255)
C_FLOOR_B = (217, 217, 217)
C_SHELF = (0, 0, 0)
C_BOX = (237, 28, 36)
C_ROBOT = (112, 48, 160)
C_DROP = (168, 230, 29)
C_PATH = (180, 130, 230)
C_TEXT = (230, 230, 235)
C_DIM = (140, 145, 155)

Cell = Tuple[int, int]


# =============================================================================
# 2. ENUMS & MESSAGES  (ROS2: these become .msg definitions)
# =============================================================================

class ItemState(Enum):
    ON_SHELF = auto()
    CARRIED = auto()
    DELIVERED = auto()


class RobotState(Enum):
    IDLE = auto()
    TO_PICK = auto()
    TO_DROP = auto()
    PARKING = auto()      # moving to a parking spot (still allowed to bid)


class Kind(Enum):
    """Message category -> decides text colour of text boxes."""
    SERVER = auto()
    BID = auto()
    WIN = auto()
    PICK = auto()
    DROP = auto()
    WARN = auto()
    INFO = auto()


KIND_COLOR = {
    Kind.SERVER: (255, 214, 10),
    Kind.BID: (255, 150, 50),
    Kind.WIN: (90, 230, 120),
    Kind.PICK: (80, 200, 255),
    Kind.DROP: (120, 160, 255),
    Kind.WARN: (255, 90, 90),
    Kind.INFO: (170, 175, 185),
}


class Topic:
    TASK = "/task_announce"      # server  -> all robots
    BID = "/bids"                # robot   -> all robots
    WINNER = "/winner"           # winner  -> everyone (server listens too)
    PICKED = "/item_picked"      # robot   -> server
    DROPPED = "/item_dropped"    # robot   -> server
    INTENT = "/robot_intent"     # robot   -> all robots (collision avoidance)
    YIELD = "/yield_request"     # robot   -> a specific robot


@dataclass
class TaskAnnounce:
    task_id: int
    item_id: int
    cell: Cell


@dataclass
class Bid:
    task_id: int
    robot_id: int
    cost: int
    access_cell: Cell        # cell next to the shelf the bidder would pick from


@dataclass
class Winner:
    task_id: int
    robot_id: int
    item_id: int
    cost: int


@dataclass
class ItemPicked:
    robot_id: int
    item_id: int
    cell: Cell


@dataclass
class ItemDropped:
    robot_id: int
    item_id: int
    cell: Cell


@dataclass
class RobotIntent:
    robot_id: int
    cell: Cell
    next_cell: Optional[Cell]
    priority: Tuple[int, int]        # lower tuple = higher priority


@dataclass
class YieldRequest:
    from_id: int
    to_id: int
    priority: Tuple[int, int]
    from_cell: Cell
    path: List[Cell]                 # requester's remaining path (don't park on it)


# =============================================================================
# 3. INFRASTRUCTURE: message bus + clock
# =============================================================================

class Clock:
    """Simulation time. (ROS2: node.get_clock().now())"""

    def __init__(self):
        self.now = 0.0


class MessageBus:
    """Tiny synchronous publish/subscribe bus. (ROS2: topics)"""

    def __init__(self, echo: bool = False):
        self._subs: Dict[str, List[Callable]] = defaultdict(list)
        self.echo = echo             # True -> print every message to the console

    def subscribe(self, topic: str, callback: Callable) -> None:
        self._subs[topic].append(callback)

    def publish(self, topic: str, msg) -> None:
        if self.echo and topic != Topic.INTENT:
            print(f"[{topic}] {msg}")
        for cb in list(self._subs[topic]):
            cb(msg)


# =============================================================================
# 4. WAREHOUSE (world state) + path planning
# =============================================================================

@dataclass
class Item:
    item_id: int
    cell: Cell
    state: ItemState = ItemState.ON_SHELF


class Warehouse:
    """Static map + item bookkeeping. Parsed from the ASCII LAYOUT."""

    def __init__(self, layout: List[str]):
        if len({len(r) for r in layout}) != 1:
            raise ValueError("All LAYOUT rows must have the same length")
        self.rows = len(layout)
        self.cols = len(layout[0])
        self.shelves: Set[Cell] = set()
        self.dropoffs: List[Cell] = []
        self.robot_starts: List[Cell] = []
        self.items: Dict[int, Item] = {}
        self.delivered = 0

        next_item_id = 1
        for y, row in enumerate(layout):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.shelves.add((x, y))
                elif ch == "B":
                    self.shelves.add((x, y))
                    self.items[next_item_id] = Item(next_item_id, (x, y))
                    next_item_id += 1
                elif ch == "R":
                    self.robot_starts.append((x, y))
                elif ch == "D":
                    self.dropoffs.append((x, y))

    # --- map queries ---
    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.cols and 0 <= c[1] < self.rows

    def is_walkable(self, c: Cell) -> bool:
        return self.in_bounds(c) and c not in self.shelves

    def neighbours(self, c: Cell) -> List[Cell]:
        x, y = c
        out = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if self.is_walkable(n):
                out.append(n)
        return out

    def access_cells(self, item_cell: Cell) -> Set[Cell]:
        """Floor cells from which a shelf cell can be reached (4-neighbourhood)."""
        return set(self.neighbours(item_cell))

    # --- world actions (ROS2: replaced by real manipulation) ---
    def pick_item(self, item_id: int) -> None:
        self.items[item_id].state = ItemState.CARRIED

    def deliver_item(self, item_id: int) -> None:
        self.items[item_id].state = ItemState.DELIVERED
        self.delivered += 1

    def count(self, state: ItemState) -> int:
        return sum(1 for i in self.items.values() if i.state == state)


def astar(wh: Warehouse, start: Cell, goals: Set[Cell],
          blocked: Set[Cell] = frozenset()) -> Optional[List[Cell]]:
    """
    A* on the 4-connected grid (Manhattan heuristic).
    `goals` may hold several cells -> the cheapest one is reached.
    `blocked` = temporary obstacles (e.g. other robots); goal cells are exempt so a
    robot can still plan *to* a cell currently occupied.
    Returns the path WITHOUT the start cell ([] if already there), None if unreachable.
    """
    goals = set(goals)
    if not goals:
        return None
    if start in goals:
        return []

    def h(c: Cell) -> int:
        return min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goals)

    tie = itertools.count()
    heap = [(h(start), 0, next(tie), start)]
    came: Dict[Cell, Cell] = {}
    best_g = {start: 0}
    while heap:
        _, g, _, cur = heapq.heappop(heap)
        if cur in goals:
            path = []
            while cur != start:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        if g > best_g.get(cur, 1 << 30):
            continue
        for nb in wh.neighbours(cur):
            if nb in blocked and nb not in goals:
                continue
            ng = g + 1
            if ng < best_g.get(nb, 1 << 30):
                best_g[nb] = ng
                came[nb] = cur
                heapq.heappush(heap, (ng + h(nb), ng, next(tie), nb))
    return None


def bfs_nearest(wh: Warehouse, start: Cell, accept: Callable[[Cell], bool],
                blocked: Set[Cell] = frozenset()) -> Optional[List[Cell]]:
    """Shortest path to the nearest cell satisfying `accept` (never crosses `blocked`)."""
    came: Dict[Cell, Optional[Cell]] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur != start and accept(cur):
            path = []
            while cur != start:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        for nb in wh.neighbours(cur):
            if nb in came or nb in blocked:
                continue
            came[nb] = cur
            q.append(nb)
    return None


# =============================================================================
# 5. TEXT BOXES (data only - drawn by the Renderer)
# =============================================================================

@dataclass
class Bubble:
    text: str
    kind: Kind
    expires: float


class MessageFeed:
    """Scrolling log shown in the top text box (server + winner + ack messages)."""

    def __init__(self, clock: Clock, max_lines: int = 6):
        self.clock = clock
        self.lines: Deque[Tuple[float, str, Kind]] = deque(maxlen=max_lines)

    def add(self, text: str, kind: Kind = Kind.SERVER) -> None:
        self.lines.append((self.clock.now, text, kind))


# =============================================================================
# 6. CENTRAL SERVER
# =============================================================================

class CentralServer:
    """
    Only job: broadcast the location of the next item to pick.
    It never chooses a robot. It merely listens to /winner (to know when to announce
    the next item) and to /item_picked, /item_dropped (acknowledgements shown on screen).
    """

    def __init__(self, bus: MessageBus, clock: Clock, wh: Warehouse, feed: MessageFeed):
        self.bus, self.clock, self.wh, self.feed = bus, clock, wh, feed
        self.pending: Deque[int] = deque(wh.items.keys())   # item ids still to be picked
        self.task_id = 0
        self.current: Optional[TaskAnnounce] = None         # task waiting for a winner
        self.announced_at = 0.0
        self.next_announce_at = 1.0
        bus.subscribe(Topic.WINNER, self.on_winner)
        bus.subscribe(Topic.PICKED, self.on_picked)
        bus.subscribe(Topic.DROPPED, self.on_dropped)

    def update(self) -> None:
        now = self.clock.now
        if self.current is None:
            if self.pending and now >= self.next_announce_at:
                self._announce()
        elif now - self.announced_at > CFP_TIMEOUT:
            self.feed.add(f"Task {self.current.task_id}: no bids received - will retry", Kind.WARN)
            self.current = None
            self.next_announce_at = now + RETRY_DELAY

    def _announce(self) -> None:
        item_id = self.pending[0]
        item = self.wh.items[item_id]
        self.task_id += 1
        self.current = TaskAnnounce(self.task_id, item_id, item.cell)
        self.announced_at = self.clock.now
        self.feed.add(f"SERVER: Task {self.task_id} - pick up item #{item_id} at {item.cell}", Kind.SERVER)
        self.bus.publish(Topic.TASK, self.current)

    def on_winner(self, msg: Winner) -> None:
        if self.current and msg.task_id == self.current.task_id:
            self.feed.add(f"WINNER: Robot {msg.robot_id} takes item #{msg.item_id} (bid {msg.cost})", Kind.WIN)
            self.pending.popleft()
            self.current = None
            self.next_announce_at = self.clock.now + TASK_GAP

    def on_picked(self, msg: ItemPicked) -> None:
        self.feed.add(f"ACK: Robot {msg.robot_id} picked item #{msg.item_id} at {msg.cell}", Kind.PICK)

    def on_dropped(self, msg: ItemDropped) -> None:
        self.feed.add(f"ACK: Robot {msg.robot_id} dropped item #{msg.item_id} at {msg.cell}", Kind.DROP)


# =============================================================================
# 7. ROBOT
# =============================================================================

@dataclass
class Auction:
    """A robot's own participation in one CNP round."""
    task_id: int
    item_id: int
    my_bid: Bid
    deadline: float


class Robot:
    MOVING_STATES = (RobotState.TO_PICK, RobotState.TO_DROP, RobotState.PARKING)

    def __init__(self, robot_id: int, cell: Cell, bus: MessageBus, clock: Clock, wh: Warehouse):
        self.id = robot_id
        self.cell = cell
        self.prev_cell = cell                 # for smooth drawing only
        self.bus, self.clock, self.wh = bus, clock, wh

        self.state = RobotState.IDLE
        self.goal: Optional[Cell] = None
        self.path: List[Cell] = []
        self.intent: Optional[Cell] = None
        self.task_item: Optional[int] = None
        self.carrying: Optional[int] = None

        self.auction: Optional[Auction] = None
        self.bid_book: Dict[int, Dict[int, Bid]] = {}    # task_id -> {robot_id: Bid}
        self.peers: Dict[int, RobotIntent] = {}          # latest intent of every other robot

        self.blocked_ticks = 0
        self.hold = 0
        self.evading = False
        self.bubbles: List[Bubble] = []

        bus.subscribe(Topic.TASK, self.on_task)
        bus.subscribe(Topic.BID, self.on_bid)
        bus.subscribe(Topic.INTENT, self.on_intent)
        bus.subscribe(Topic.YIELD, self.on_yield)

    # ------------------------------------------------------------------ helpers
    def say(self, text: str, kind: Kind = Kind.INFO, ttl: float = BUBBLE_TTL) -> None:
        self.bubbles.append(Bubble(text, kind, self.clock.now + ttl))
        self.bubbles = self.bubbles[-3:]

    def priority(self) -> Tuple[int, int]:
        """Lower = more important. A robot standing on a dropoff must be able to leave."""
        return (0 if self.cell in self.wh.dropoffs else 1, self.id)

    def _peer_cells(self) -> Set[Cell]:
        return {p.cell for p in self.peers.values()}

    def _is_available(self) -> bool:
        return (self.state in (RobotState.IDLE, RobotState.PARKING)
                and self.carrying is None and self.auction is None and not self.evading)

    def _plan(self, goals: Set[Cell]) -> List[Cell]:
        """A* avoiding other robots if possible, otherwise ignoring them."""
        path = astar(self.wh, self.cell, goals, self._peer_cells())
        if path is None:
            path = astar(self.wh, self.cell, goals)
        return path if path is not None else []

    # ------------------------------------------------------- CNP: bidding phase
    def on_task(self, msg: TaskAnnounce) -> None:
        if not self._is_available():
            self.say("Busy - no bid", Kind.INFO, 1.2)
            return
        path = astar(self.wh, self.cell, self.wh.access_cells(msg.cell))
        if path is None:
            self.say("No path - no bid", Kind.WARN, 1.5)
            return
        access = path[-1] if path else self.cell
        bid = Bid(msg.task_id, self.id, len(path), access)
        self.auction = Auction(msg.task_id, msg.item_id, bid, self.clock.now + BID_WINDOW)
        self.say(f"Bid: {bid.cost}", Kind.BID, BID_WINDOW + 0.8)
        self.bus.publish(Topic.BID, bid)

    def on_bid(self, bid: Bid) -> None:
        # Always record - another robot's bid can arrive before we even got the task.
        self.bid_book.setdefault(bid.task_id, {})[bid.robot_id] = bid
        for old in [t for t in self.bid_book if t < bid.task_id - 3]:
            del self.bid_book[old]

    def update(self) -> None:
        """Called every frame: close the auction when the bid window expires."""
        a = self.auction
        if a is None or self.clock.now < a.deadline:
            return
        self.auction = None
        bids = self.bid_book.get(a.task_id, {})
        win = min(bids.values(), key=lambda b: (b.cost, b.robot_id))   # same result on every robot
        if win.robot_id == self.id:
            self._win(a)
        else:
            self.say(f"Lost to R{win.robot_id}", Kind.INFO, 1.5)

    def _win(self, a: Auction) -> None:
        self.state = RobotState.TO_PICK
        self.task_item = a.item_id
        self.goal = a.my_bid.access_cell
        self.path = self._plan({self.goal})
        self.evading, self.hold, self.blocked_ticks = False, 0, 0
        self.say(f"I WIN item #{a.item_id} (bid {a.my_bid.cost})", Kind.WIN)
        self.bus.publish(Topic.WINNER, Winner(a.task_id, self.id, a.item_id, a.my_bid.cost))

    # ----------------------------------------------- movement: two-phase tick
    def plan_step(self) -> None:
        """Phase 1: decide + broadcast where we want to go this tick."""
        self.intent = None
        if self.auction is None and self.hold == 0:
            if not self.path and self.state in self.MOVING_STATES and self.cell != self.goal:
                self.path = self._plan({self.goal})
            if self.path:
                self.intent = self.path[0]
        self.bus.publish(Topic.INTENT, RobotIntent(self.id, self.cell, self.intent, self.priority()))

    def execute_step(self) -> None:
        """Phase 2: move if the target cell is safe, otherwise handle being blocked."""
        self.prev_cell = self.cell
        if self.hold > 0:
            self.hold -= 1
            return
        if self.auction is not None:
            return
        if self.intent is None:
            if self.state in self.MOVING_STATES and self.cell == self.goal and not self.evading:
                self._on_arrive()
            return
        if self._can_move(self.intent):
            self.cell = self.path.pop(0)
            self.blocked_ticks = 0
            if self.evading and not self.path:
                self.evading = False
                self.hold = HOLD_TICKS       # give the requester time to pass
        else:
            self._on_blocked(self.intent)

    def on_intent(self, msg: RobotIntent) -> None:
        if msg.robot_id != self.id:
            self.peers[msg.robot_id] = msg

    def _can_move(self, target: Cell) -> bool:
        """Collision rule: target must be empty and not wanted by a higher-priority robot."""
        me = self.priority()
        for p in self.peers.values():
            if p.cell == target:
                return False
            if p.next_cell == target and p.priority < me:
                return False
        return True

    def _on_blocked(self, target: Cell) -> None:
        self.blocked_ticks += 1
        if self.blocked_ticks < REPLAN_AFTER or self.goal is None and not self.evading:
            return
        # 1) try a detour around everybody
        if self.goal is not None and not self.evading:
            alt = astar(self.wh, self.cell, {self.goal}, self._peer_cells())
            if alt is not None and alt != self.path and len(alt) <= len(self.path) + DETOUR_LIMIT:
                self.path = alt
                self.blocked_ticks = 0
                self.say("Detour", Kind.INFO, 1.2)
                return
        # 2) otherwise ask whoever is in the way to step aside
        blocker = next((p for p in self.peers.values() if p.cell == target), None)
        if blocker:
            self.bus.publish(Topic.YIELD, YieldRequest(self.id, blocker.robot_id, self.priority(),
                                                       self.cell, list(self.path)))

    def on_yield(self, req: YieldRequest) -> None:
        if req.to_id != self.id or self.evading or self.auction is not None or self.hold > 0:
            return
        idle = self.state in (RobotState.IDLE, RobotState.PARKING)
        if idle or (self.blocked_ticks > 0 and req.priority < self.priority()):
            self._evade(req)

    def _evade(self, req: YieldRequest) -> None:
        keep_out = set(req.path) | {req.from_cell}
        path = bfs_nearest(self.wh, self.cell,
                           lambda c: c not in keep_out and c not in self.wh.dropoffs,
                           self._peer_cells())
        if not path:
            return
        if self.state in (RobotState.IDLE, RobotState.PARKING):
            self.state, self.goal, self.path = RobotState.PARKING, path[-1], path    # new parking spot
        else:
            self.path, self.evading = path, True        # keep the task, resume after the hold
        self.blocked_ticks = 0
        self.say(f"Yielding to R{req.from_id}", Kind.INFO, 1.5)

    # --------------------------------------------------------------- arrivals
    def _on_arrive(self) -> None:
        if self.state == RobotState.TO_PICK:
            self._pick()
        elif self.state == RobotState.TO_DROP:
            self._drop()
        elif self.state == RobotState.PARKING:
            self.state, self.goal = RobotState.IDLE, None

    def _pick(self) -> None:
        item = self.wh.items[self.task_item]
        self.wh.pick_item(item.item_id)
        self.carrying = item.item_id
        self.say(f"Item #{item.item_id} picked -> server", Kind.PICK)
        self.bus.publish(Topic.PICKED, ItemPicked(self.id, item.item_id, item.cell))
        # head for the closest dropoff
        best = None
        for d in self.wh.dropoffs:
            p = astar(self.wh, self.cell, {d})
            if p is not None and (best is None or len(p) < len(best[1])):
                best = (d, p)
        self.state = RobotState.TO_DROP
        self.goal = best[0] if best else None
        self.path = self._plan({self.goal}) if best else []

    def _drop(self) -> None:
        item_id = self.carrying
        self.wh.deliver_item(item_id)
        self.carrying = self.task_item = None
        self.say(f"Item #{item_id} dropped -> server", Kind.DROP)
        self.bus.publish(Topic.DROPPED, ItemDropped(self.id, item_id, self.cell))
        self._go_park()

    def _go_park(self) -> None:
        """Leave the dropoff and wait somewhere that doesn't block it."""
        def ok(c: Cell) -> bool:
            far = all(abs(c[0] - d[0]) + abs(c[1] - d[1]) >= PARK_MIN_DIST for d in self.wh.dropoffs)
            return far and c not in self._peer_cells()
        path = bfs_nearest(self.wh, self.cell, ok, self._peer_cells())
        if path:
            self.state, self.goal, self.path = RobotState.PARKING, path[-1], path
        else:
            self.state, self.goal, self.path = RobotState.IDLE, None, []


# =============================================================================
# 8. SIMULATION (owns everything, advances time)
# =============================================================================

class Simulation:
    def __init__(self, layout: List[str], echo: bool = False):
        self.clock = Clock()
        self.bus = MessageBus(echo)
        self.warehouse = Warehouse(layout)
        self.feed = MessageFeed(self.clock)
        self.robots: List[Robot] = [
            Robot(i + 1, start, self.bus, self.clock, self.warehouse)
            for i, start in enumerate(self.warehouse.robot_starts)
        ]
        self.server = CentralServer(self.bus, self.clock, self.warehouse, self.feed)
        self.tick_timer = 0.0
        self.speed = 1.0
        self.paused = False

    @property
    def tick_progress(self) -> float:
        return min(1.0, self.tick_timer / STEP_INTERVAL)

    def update(self, dt: float) -> None:
        if self.paused:
            return
        dt *= self.speed
        self.clock.now += dt
        self.server.update()
        for r in self.robots:
            r.update()
        self.tick_timer += dt
        while self.tick_timer >= STEP_INTERVAL:
            self.tick_timer -= STEP_INTERVAL
            self._tick()

    def _tick(self) -> None:
        for r in self.robots:      # phase 1: everybody announces intent
            r.plan_step()
        for r in self.robots:      # phase 2: everybody moves (or not)
            r.execute_step()

    def all_done(self) -> bool:
        return self.warehouse.delivered == len(self.warehouse.items)


# =============================================================================
# 9. RENDERER (the only part that needs pygame - replace with RViz in ROS2)
# =============================================================================

class Renderer:
    def __init__(self, screen, sim: Simulation):
        self.screen, self.sim = screen, sim
        self.wh = sim.warehouse
        self.show_paths = True
        self.gx0 = MARGIN
        self.gy0 = TOP_PANEL_H + 2 * MARGIN
        self.gw = self.wh.cols * CELL
        self.gh = self.wh.rows * CELL
        self.f_title = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 16, bold=True)
        self.f_text = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 14)
        self.f_small = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 12, bold=True)

    @staticmethod
    def window_size(wh: Warehouse) -> Tuple[int, int]:
        w = MARGIN + wh.cols * CELL + MARGIN + SIDE_PANEL_W + MARGIN
        h = TOP_PANEL_H + 2 * MARGIN + wh.rows * CELL + MARGIN
        return w, h

    # --- coordinate helpers ---
    def cell_xy(self, c) -> Tuple[float, float]:
        return self.gx0 + c[0] * CELL, self.gy0 + c[1] * CELL

    def robot_xy(self, r: Robot) -> Tuple[float, float]:
        t = self.sim.tick_progress
        x0, y0 = self.cell_xy(r.prev_cell)
        x1, y1 = self.cell_xy(r.cell)
        return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t

    def text(self, s, x, y, color=C_TEXT, font=None):
        surf = (font or self.f_text).render(s, True, color)
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    # --- main draw ---
    def draw(self) -> None:
        self.screen.fill(C_BG)
        self.draw_top_panel()
        self.draw_grid()
        self.draw_items()
        if self.show_paths:
            self.draw_paths()
        self.draw_robots()
        self.draw_bubbles()
        self.draw_side_panel()

    def draw_top_panel(self) -> None:
        w = self.screen.get_width() - 2 * MARGIN
        rect = pygame.Rect(MARGIN, MARGIN, w, TOP_PANEL_H)
        pygame.draw.rect(self.screen, (20, 20, 20), rect, border_radius=8)
        # pygame.draw.rect(self.screen, KIND_COLOR[Kind.SERVER], rect, 2, border_radius=8)
        self.text("CENTRAL SERVER / WINNER FEED", rect.x + 12, rect.y + 8, KIND_COLOR[Kind.SERVER], self.f_title)
        lines = list(self.sim.feed.lines)
        for i, (_, txt, kind) in enumerate(lines):
            newest = i == len(lines) - 1
            col = KIND_COLOR[kind] if newest else tuple(int(c * 0.7) for c in KIND_COLOR[kind])
            self.text(txt, rect.x + 14, rect.y + 34 + i * 18, col)

    def draw_grid(self) -> None:
        for y in range(self.wh.rows):
            for x in range(self.wh.cols):
                px, py = self.cell_xy((x, y))
                col = C_FLOOR_A if (x + y) % 2 == 0 else C_FLOOR_B
                if (x, y) in self.wh.shelves:
                    col = C_SHELF
                elif (x, y) in self.wh.dropoffs:
                    col = C_DROP
                pygame.draw.rect(self.screen, col, (px, py, CELL, CELL))
        for d in self.wh.dropoffs:
            px, py = self.cell_xy(d)
            self.text("DROP", px + 8, py + 6, (40, 60, 0), self.f_small)
            self.text(str(self.wh.delivered), px + CELL // 2 - 4, py + CELL - 22, (40, 60, 0), self.f_title)

    def draw_items(self) -> None:
        active = self.sim.server.current
        for item in self.wh.items.values():
            if item.state != ItemState.ON_SHELF:
                continue
            px, py = self.cell_xy(item.cell)
            pygame.draw.rect(self.screen, C_BOX, (px + 6, py + 6, CELL - 12, CELL - 12))
            self.text(f"#{item.item_id}", px + 14, py + CELL // 2 - 7, (255, 255, 255), self.f_small)
            if active and active.item_id == item.item_id:          # item currently being auctioned
                pygame.draw.rect(self.screen, KIND_COLOR[Kind.SERVER], (px + 1, py + 1, CELL - 2, CELL - 2), 3)

    def draw_paths(self) -> None:
        half = CELL // 2
        for r in self.sim.robots:
            if not r.path:
                continue
            pts = [(self.cell_xy(r.cell)[0] + half, self.cell_xy(r.cell)[1] + half)]
            pts += [(self.cell_xy(c)[0] + half, self.cell_xy(c)[1] + half) for c in r.path]
            pygame.draw.lines(self.screen, C_PATH, False, pts, 3)

    def draw_robots(self) -> None:
        for r in self.sim.robots:
            x, y = self.robot_xy(r)
            body = pygame.Rect(int(x) + 8, int(y) + 8, CELL - 16, CELL - 16)
            pygame.draw.rect(self.screen, C_ROBOT, body, border_radius=8)
            pygame.draw.rect(self.screen, (255, 255, 255), body, 2, border_radius=8)
            label = self.f_title.render(str(r.id), True, (255, 255, 255))
            self.screen.blit(label, label.get_rect(center=body.center))
            if r.carrying is not None:
                pygame.draw.rect(self.screen, C_BOX, (body.right - 14, body.y - 4, 16, 16))
                pygame.draw.rect(self.screen, (255, 255, 255), (body.right - 14, body.y - 4, 16, 16), 1)

    def draw_bubbles(self) -> None:
        now = self.sim.clock.now
        h, gap = 18, 2
        for r in self.sim.robots:
            r.bubbles = [b for b in r.bubbles if b.expires > now]
            if not r.bubbles:
                continue
            x, y = self.robot_xy(r)
            total = len(r.bubbles) * (h + gap)
            above = y - 4 - total >= self.gy0 - MARGIN
            for i, b in enumerate(reversed(r.bubbles)):           # newest closest to robot
                surf = self.f_small.render(b.text, True, (255, 255, 255))
                w = surf.get_width() + 10
                bx = max(self.gx0, min(x + CELL // 2 - w // 2, self.gx0 + self.gw - w))
                by = y - 4 - (i + 1) * (h + gap) + gap if above else y + CELL + 4 + i * (h + gap)
                box = pygame.Rect(int(bx), int(by), w, h)
                pygame.draw.rect(self.screen, (25, 25, 32), box, border_radius=5)
                pygame.draw.rect(self.screen, KIND_COLOR[b.kind], box, 2, border_radius=5)
                self.screen.blit(surf, (box.x + 5, box.y + 2))

    def draw_side_panel(self) -> None:
        x = self.gx0 + self.gw + MARGIN
        y = self.gy0
        wh = self.wh
        self.text("LEGEND", x, y, C_TEXT, self.f_title); y += 24
        for col, name in ((C_SHELF, "Shelf / obstacle"), (C_BOX, "Box"), (C_ROBOT, "Robot"), (C_DROP, "Dropoff")):
            pygame.draw.rect(self.screen, col, (x, y, 16, 16))
            pygame.draw.rect(self.screen, (200, 200, 200), (x, y, 16, 16), 1)
            self.text(name, x + 24, y, C_DIM); y += 22
        y += 8
        self.text("ITEMS", x, y, C_TEXT, self.f_title); y += 24
        self.text(f"on shelf  : {wh.count(ItemState.ON_SHELF)}", x, y, C_DIM); y += 20
        self.text(f"in transit: {wh.count(ItemState.CARRIED)}", x, y, C_DIM); y += 20
        self.text(f"delivered : {wh.delivered}", x, y, C_DIM); y += 28
        self.text("ROBOTS", x, y, C_TEXT, self.f_title); y += 24
        for r in self.sim.robots:
            tag = f"#{r.carrying}" if r.carrying else ("..." if r.auction else "")
            self.text(f"R{r.id} {r.state.name:<8} {tag}", x, y, C_DIM); y += 20
        y += 14
        # self.text("CONTROLS", x, y, C_TEXT, self.f_title); y += 24
        # for line in ("SPACE pause", "UP/DOWN speed", "P show paths", "R restart", "ESC quit"):
        #     self.text(line, x, y, C_DIM); y += 20
        # status = "PAUSED" if self.sim.paused else f"speed x{self.sim.speed:.1f}"
        # if self.sim.all_done():
        #     status = "ALL ITEMS DELIVERED"
        # self.text(status, x, y + 10, KIND_COLOR[Kind.WIN] if self.sim.all_done() else C_TEXT, self.f_title)


# =============================================================================
# 10. MAIN LOOP
# =============================================================================

def main() -> None:
    pygame.init()
    sim = Simulation(LAYOUT)
    screen = pygame.display.set_mode(Renderer.window_size(sim.warehouse))
    pygame.display.set_caption("Decentralised warehouse robots - CNP + A*")
    renderer = Renderer(screen, sim)
    clock = pygame.time.Clock()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif ev.key == pygame.K_UP:
                    sim.speed = min(8.0, sim.speed * 1.5)
                elif ev.key == pygame.K_DOWN:
                    sim.speed = max(0.25, sim.speed / 1.5)
                elif ev.key == pygame.K_p:
                    renderer.show_paths = not renderer.show_paths
                elif ev.key == pygame.K_r:
                    sim = Simulation(LAYOUT)
                    renderer = Renderer(screen, sim)
        sim.update(dt)
        renderer.draw()
        pygame.display.flip()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
