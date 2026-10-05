# Motion Gesture & Event Specification

## Назначение

Этот документ описывает слой распознавания поз, жестов и действий поверх:

- MediaPipe Pose Landmarker
- MediaPipe Hand Landmarker

MediaPipe сам в основном отдаёт landmarks и связанные confidence/visibility значения.

События вроде:

- `RIGHT_ARM_RAISED`
- `PINCH_STARTED`
- `RIGHT_PUNCH`
- `SQUAT_REP`
- `JUMP`

не приходят готовыми из Pose Landmarker или Hand Landmarker. Они вычисляются приложением поверх координат, углов, расстояний, скоростей и истории нескольких кадров.

Документ является спецификацией Gesture/Event Engine.

---

# 1. Основная архитектура

```text
Camera
  ↓
Pose Landmarker ───────────┐
                           │
Hand Landmarker ───────────┤
                           ↓
                     Raw Landmarks
                           ↓
                        Smoothing
                           ↓
                     Geometry Layer
               angles / distances / vectors
                           ↓
                      Frame State
                           ↓
                       Pose Rules
                           ↓
                     Temporal Engine
                           ↓
                       Event Engine
                           ↓
                         Events
```

Игра, UI или другое приложение не должны напрямую анализировать MediaPipe landmarks.

Они должны получать готовые состояния и события.

Пример:

```python
Event(
    type=EventType.RIGHT_PUNCH,
    timestamp_ms=123456,
    confidence=0.92,
)
```

---

# 2. Три типа распознавания

Нужно различать три сущности.

## 2.1 State

Состояние сохраняется столько времени, сколько условие истинно.

Примеры:

```text
RIGHT_ARM_UP
LEFT_FIST
SQUATTING
LEAN_LEFT
```

Если человек держит руку поднятой 5 секунд:

```text
RIGHT_ARM_UP = True
```

остаётся активным всё это время.

## 2.2 Transition

Переход генерируется один раз при изменении состояния.

Пример:

```text
RIGHT_ARM_RAISED
RIGHT_ARM_LOWERED
```

Последовательность:

```text
arm down

RIGHT_ARM_RAISED
↓
RIGHT_ARM_UP
RIGHT_ARM_UP
RIGHT_ARM_UP
↓
RIGHT_ARM_LOWERED

arm down
```

## 2.3 Action

Действие определяется по последовательности кадров.

Примеры:

```text
RIGHT_PUNCH
SWIPE_LEFT
JUMP
CLAP
SQUAT_REP
```

Для action необходимо хранить историю движения.

---

# 3. MediaPipe landmarks

## 3.1 Pose Landmarker

Pose Landmarker содержит 33 landmark точки тела.

Основные точки:

```text
0   nose
11  left shoulder
12  right shoulder
13  left elbow
14  right elbow
15  left wrist
16  right wrist
17  left pinky
18  right pinky
19  left index
20  right index
21  left thumb
22  right thumb
23  left hip
24  right hip
25  left knee
26  right knee
27  left ankle
28  right ankle
29  left heel
30  right heel
31  left foot index
32  right foot index
```

## 3.2 Hand Landmarker

Каждая кисть содержит 21 landmark.

```text
0   wrist
1   thumb_cmc
2   thumb_mcp
3   thumb_ip
4   thumb_tip
5   index_mcp
6   index_pip
7   index_dip
8   index_tip
9   middle_mcp
10  middle_pip
11  middle_dip
12  middle_tip
13  ring_mcp
14  ring_pip
15  ring_dip
16  ring_tip
17  pinky_mcp
18  pinky_pip
19  pinky_dip
20  pinky_tip
```

Hand Landmarker также должен использовать handedness:

```text
LEFT
RIGHT
```

---

# 4. Базовые события трекинга

```text
PERSON_DETECTED
PERSON_LOST
PERSON_TRACKING_STABLE
PERSON_TRACKING_UNSTABLE
BODY_FULLY_VISIBLE
BODY_PARTIALLY_VISIBLE
UPPER_BODY_VISIBLE
LOWER_BODY_VISIBLE
LEFT_HAND_DETECTED
LEFT_HAND_LOST
RIGHT_HAND_DETECTED
RIGHT_HAND_LOST
BOTH_HANDS_DETECTED
BOTH_HANDS_LOST
HAND_ENTERED_FRAME
HAND_LEFT_FRAME
```

---

# 5. Голова

## States

```text
HEAD_CENTER
HEAD_LEFT
HEAD_RIGHT
HEAD_UP
HEAD_DOWN
HEAD_TILTED_LEFT
HEAD_TILTED_RIGHT
```

## Transitions / Actions

```text
HEAD_TURNED_LEFT
HEAD_TURNED_RIGHT
HEAD_NOD
HEAD_SHAKE
```

`HEAD_NOD` и `HEAD_SHAKE` требуют анализа нескольких кадров.

---

# 6. Плечи

```text
SHOULDERS_LEVEL
LEFT_SHOULDER_UP
RIGHT_SHOULDER_UP
SHOULDERS_RAISED
SHOULDERS_LOWERED
SHRUG
```

---

# 7. Левая рука

```text
LEFT_ARM_DOWN
LEFT_ARM_UP
LEFT_ARM_FORWARD
LEFT_ARM_BACKWARD
LEFT_ARM_SIDE
LEFT_ARM_BENT
LEFT_ARM_STRAIGHT
LEFT_ELBOW_UP
LEFT_ELBOW_DOWN
LEFT_HAND_ABOVE_HEAD
LEFT_HAND_ABOVE_SHOULDER
LEFT_HAND_BELOW_SHOULDER
LEFT_HAND_NEAR_FACE
LEFT_HAND_NEAR_CHEST
LEFT_HAND_NEAR_HIP
LEFT_ARM_RAISED
LEFT_ARM_LOWERED
LEFT_ARM_EXTENDED
LEFT_ARM_BENT_STARTED
```

---

# 8. Правая рука

```text
RIGHT_ARM_DOWN
RIGHT_ARM_UP
RIGHT_ARM_FORWARD
RIGHT_ARM_BACKWARD
RIGHT_ARM_SIDE
RIGHT_ARM_BENT
RIGHT_ARM_STRAIGHT
RIGHT_ELBOW_UP
RIGHT_ELBOW_DOWN
RIGHT_HAND_ABOVE_HEAD
RIGHT_HAND_ABOVE_SHOULDER
RIGHT_HAND_BELOW_SHOULDER
RIGHT_HAND_NEAR_FACE
RIGHT_HAND_NEAR_CHEST
RIGHT_HAND_NEAR_HIP
RIGHT_ARM_RAISED
RIGHT_ARM_LOWERED
RIGHT_ARM_EXTENDED
RIGHT_ARM_BENT_STARTED
```

---

# 9. Состояния обеих рук

```text
BOTH_ARMS_DOWN
BOTH_ARMS_UP
BOTH_ARMS_FORWARD
BOTH_ARMS_SIDE
BOTH_ARMS_BENT
BOTH_ARMS_STRAIGHT
HANDS_ABOVE_HEAD
HANDS_BELOW_HEAD
HANDS_TOGETHER
HANDS_APART
ARMS_CROSSED
HANDS_ON_HIPS
HANDS_BEHIND_HEAD
```

---

# 10. Классические body poses

```text
T_POSE
Y_POSE
A_POSE
X_POSE
ARMS_CROSSED_POSE
HANDS_UP_POSE
HANDS_ON_HIPS_POSE
HANDS_BEHIND_HEAD_POSE
```

При необходимости можно добавлять пользовательские pose templates.

---

# 11. Корпус

```text
TORSO_CENTER
LEAN_LEFT
LEAN_RIGHT
LEAN_FORWARD
LEAN_BACKWARD
TORSO_ROTATED_LEFT
TORSO_ROTATED_RIGHT
LEANED_LEFT
LEANED_RIGHT
TORSO_TURN_LEFT
TORSO_TURN_RIGHT
```

---

# 12. Общая поза тела

```text
STANDING
SITTING
CROUCHING
SQUATTING
KNEELING
KNEELING_LEFT
KNEELING_RIGHT
LYING
LYING_FACE_UP
LYING_FACE_DOWN
```

Некоторые состояния, особенно `LYING`, будут менее надёжны при обычной фронтальной веб-камере.

---

# 13. Ноги

```text
LEFT_LEG_STRAIGHT
LEFT_LEG_BENT
LEFT_KNEE_UP
LEFT_LEG_RAISED
LEFT_FOOT_FORWARD
LEFT_FOOT_BACKWARD
LEFT_FOOT_OFF_GROUND
RIGHT_LEG_STRAIGHT
RIGHT_LEG_BENT
RIGHT_KNEE_UP
RIGHT_LEG_RAISED
RIGHT_FOOT_FORWARD
RIGHT_FOOT_BACKWARD
RIGHT_FOOT_OFF_GROUND
LEGS_APART
LEGS_TOGETHER
FEET_APART
FEET_TOGETHER
BOTH_FEET_ON_GROUND
BOTH_FEET_OFF_GROUND
```

---

# 14. Finger states

Для каждой руки каждый палец должен иметь состояние `EXTENDED` или `FOLDED`.

```text
LEFT_THUMB_EXTENDED
LEFT_THUMB_FOLDED
LEFT_INDEX_EXTENDED
LEFT_INDEX_FOLDED
LEFT_MIDDLE_EXTENDED
LEFT_MIDDLE_FOLDED
LEFT_RING_EXTENDED
LEFT_RING_FOLDED
LEFT_PINKY_EXTENDED
LEFT_PINKY_FOLDED
RIGHT_THUMB_EXTENDED
RIGHT_THUMB_FOLDED
RIGHT_INDEX_EXTENDED
RIGHT_INDEX_FOLDED
RIGHT_MIDDLE_EXTENDED
RIGHT_MIDDLE_FOLDED
RIGHT_RING_EXTENDED
RIGHT_RING_FOLDED
RIGHT_PINKY_EXTENDED
RIGHT_PINKY_FOLDED
```

---

# 15. Базовые формы кисти

Для обеих рук поддержать:

```text
OPEN_PALM
FIST
POINT_INDEX
THUMBS_UP
THUMBS_DOWN
PEACE
VICTORY
OK_SIGN
ROCK_SIGN
CALL_ME
ONE_FINGER
TWO_FINGERS
THREE_FINGERS
FOUR_FINGERS
FIVE_FINGERS
```

Сторона кодируется отдельно, например:

```text
LEFT_OPEN_PALM
RIGHT_OPEN_PALM
LEFT_FIST
RIGHT_FIST
```

---

# 16. Finger touch

```text
THUMB_INDEX_TOUCH
THUMB_MIDDLE_TOUCH
THUMB_RING_TOUCH
THUMB_PINKY_TOUCH
LEFT_THUMB_INDEX_TOUCH
RIGHT_THUMB_INDEX_TOUCH
```

и аналогично для остальных пальцев.

---

# 17. Pinch

Основной pinch:

```text
thumb_tip ↔ index_tip
```

События:

```text
LEFT_PINCH_STARTED
LEFT_PINCH_ACTIVE
LEFT_PINCH_MOVED
LEFT_PINCH_RELEASED
LEFT_PINCH_CANCELLED
RIGHT_PINCH_STARTED
RIGHT_PINCH_ACTIVE
RIGHT_PINCH_MOVED
RIGHT_PINCH_RELEASED
RIGHT_PINCH_CANCELLED
```

Дополнительно:

```text
PINCH_INDEX
PINCH_MIDDLE
PINCH_RING
PINCH_PINKY
```

---

# 18. Grab

```text
LEFT_GRAB_STARTED
LEFT_GRAB_ACTIVE
LEFT_GRAB_RELEASED
LEFT_GRAB_CANCELLED
RIGHT_GRAB_STARTED
RIGHT_GRAB_ACTIVE
RIGHT_GRAB_RELEASED
RIGHT_GRAB_CANCELLED
LEFT_GRAB
RIGHT_GRAB
```

Grab можно определять по закрыванию большинства пальцев.

---

# 19. Pointing

```text
LEFT_POINT
RIGHT_POINT
LEFT_POINT_UP
LEFT_POINT_DOWN
LEFT_POINT_LEFT
LEFT_POINT_RIGHT
LEFT_POINT_FORWARD
RIGHT_POINT_UP
RIGHT_POINT_DOWN
RIGHT_POINT_LEFT
RIGHT_POINT_RIGHT
RIGHT_POINT_FORWARD
POINT_STARTED
POINT_ACTIVE
POINT_MOVED
POINT_ENDED
POINT_CANCELLED
```

---

# 20. Swipe

```text
LEFT_SWIPE_LEFT
LEFT_SWIPE_RIGHT
LEFT_SWIPE_UP
LEFT_SWIPE_DOWN
RIGHT_SWIPE_LEFT
RIGHT_SWIPE_RIGHT
RIGHT_SWIPE_UP
RIGHT_SWIPE_DOWN
SWIPE_LEFT
SWIPE_RIGHT
SWIPE_UP
SWIPE_DOWN
```

Swipe требует анализа траектории кисти.

---

# 21. Wave

```text
LEFT_WAVE
RIGHT_WAVE
WAVE_STARTED
WAVE_ACTIVE
WAVE_ENDED
```

Wave определяется как несколько смен направления движения кисти за короткий интервал.

---

# 22. Punch

```text
LEFT_PUNCH
RIGHT_PUNCH
LEFT_PUNCH_FORWARD
RIGHT_PUNCH_FORWARD
LEFT_PUNCH_LEFT
RIGHT_PUNCH_RIGHT
LEFT_UPPERCUT
RIGHT_UPPERCUT
LEFT_HOOK
RIGHT_HOOK
```

Punch должен учитывать направление движения, скорость запястья, положение локтя, изменение длины руки и cooldown.

---

# 23. Push

```text
LEFT_PUSH
RIGHT_PUSH
BOTH_HANDS_PUSH
```

Push отличается от punch меньшей резкостью и часто открытой ладонью.

---

# 24. Pull

```text
LEFT_PULL
RIGHT_PULL
BOTH_HANDS_PULL
```

---

# 25. Circular hand motion

```text
LEFT_HAND_CIRCLE_CLOCKWISE
LEFT_HAND_CIRCLE_COUNTERCLOCKWISE
RIGHT_HAND_CIRCLE_CLOCKWISE
RIGHT_HAND_CIRCLE_COUNTERCLOCKWISE
```

---

# 26. Clap

```text
HANDS_NEAR
HANDS_TOUCHING
CLAP
DOUBLE_CLAP
```

---

# 27. Hands together / apart

```text
HANDS_MOVED_TOGETHER
HANDS_MOVED_APART
```

Можно использовать для zoom/scale UI-команд.

---

# 28. Throw

```text
LEFT_THROW
RIGHT_THROW
```

Возможная последовательность:

```text
arm back
↓
rapid forward motion
↓
release-like hand state
```

---

# 29. Catch

```text
LEFT_CATCH
RIGHT_CATCH
BOTH_HANDS_CATCH
```

Обычно это составное событие и лучше использовать при наличии игрового объекта.

---

# 30. Jump

```text
JUMP_STARTED
JUMP
AIRBORNE
LAND
```

State machine:

```text
GROUND
↓
JUMP_STARTED
↓
AIRBORNE
↓
LAND
↓
GROUND
```

---

# 31. Squat

```text
STANDING
SQUATTING
SQUAT_STARTED
SQUAT_DOWN
SQUAT_BOTTOM
SQUAT_UP
SQUAT_COMPLETED
SQUAT_REP
```

State machine:

```text
STANDING
↓
GOING_DOWN
↓
BOTTOM
↓
GOING_UP
↓
STANDING
↓
SQUAT_REP
```

---

# 32. Lunge

```text
LEFT_LUNGE_STARTED
LEFT_LUNGE
LEFT_LUNGE_COMPLETED
RIGHT_LUNGE_STARTED
RIGHT_LUNGE
RIGHT_LUNGE_COMPLETED
```

---

# 33. Steps

```text
STEP_LEFT
STEP_RIGHT
STEP_FORWARD
STEP_BACKWARD
LEFT_STEP_STARTED
RIGHT_STEP_STARTED
```

---

# 34. Walking

```text
WALKING
NOT_WALKING
WALK_STARTED
WALK_STOPPED
```

Обычная stationary webcam не всегда позволяет надёжно распознавать реальное перемещение по комнате.

---

# 35. Running

```text
RUNNING
RUN_STARTED
RUN_STOPPED
RUNNING_IN_PLACE
```

---

# 36. Jumping Jack

```text
JUMPING_JACK_STARTED
JUMPING_JACK_OPEN
JUMPING_JACK_CLOSED
JUMPING_JACK_REP
```

---

# 37. Dodge

```text
DODGE_LEFT
DODGE_RIGHT
DODGE_DOWN
DUCK
```

---

# 38. Kick

```text
LEFT_KICK
RIGHT_KICK
LEFT_KICK_FORWARD
RIGHT_KICK_FORWARD
LEFT_KICK_SIDE
RIGHT_KICK_SIDE
LEFT_HIGH_KICK
RIGHT_HIGH_KICK
```

---

# 39. Block / Defense

```text
BLOCK_LEFT
BLOCK_RIGHT
BLOCK_HIGH
BLOCK_LOW
BLOCK_CENTER
GUARD_POSE
```

---

# 40. Combat events

```text
LEFT_JAB
RIGHT_JAB
LEFT_CROSS
RIGHT_CROSS
LEFT_HOOK
RIGHT_HOOK
LEFT_UPPERCUT
RIGHT_UPPERCUT
LEFT_BLOCK
RIGHT_BLOCK
DODGE_LEFT
DODGE_RIGHT
DUCK
```

Не реализовывать все combat events на первом этапе.

---

# 41. Fitness events

```text
SQUAT_REP
LUNGE_LEFT_REP
LUNGE_RIGHT_REP
JUMPING_JACK_REP
HIGH_KNEE_LEFT
HIGH_KNEE_RIGHT
HIGH_KNEE_REP
ARM_RAISE_REP
SIDE_BEND_LEFT
SIDE_BEND_RIGHT
TOE_TOUCH_LEFT
TOE_TOUCH_RIGHT
```

---

# 42. UI events

## Cursor

```text
CURSOR_ACTIVE
CURSOR_MOVED
CURSOR_LOST
```

Координата курсора может вычисляться по `index_tip` или `wrist`.

## Click

```text
POINTER_DOWN
POINTER_UP
CLICK
DOUBLE_CLICK
```

Например:

```text
PINCH_STARTED → POINTER_DOWN
PINCH_RELEASED → POINTER_UP
```

## Drag

```text
DRAG_STARTED
DRAG
DRAG_ENDED
DRAG_CANCELLED
```

---

# 43. Two-hand UI events

```text
TWO_HAND_INTERACTION_STARTED
TWO_HAND_MOVE
TWO_HAND_ROTATE
TWO_HAND_SCALE
TWO_HAND_INTERACTION_ENDED
TWO_HAND_INTERACTION_CANCELLED
ZOOM_IN
ZOOM_OUT
```

---

# 44. Navigation events

Semantic events:

```text
SELECT
CONFIRM
CANCEL
BACK
NEXT
PREVIOUS
MENU
PAUSE
RESUME
ZOOM_IN
ZOOM_OUT
ROTATE_LEFT
ROTATE_RIGHT
```

Semantic events не должны распознаваться напрямую.

Пример mapping:

```text
RIGHT_SWIPE_RIGHT → NEXT
RIGHT_PINCH_STARTED → SELECT
```

---

# 45. Object interaction

```text
GRAB_OBJECT
HOLD_OBJECT
MOVE_OBJECT
DROP_OBJECT
THROW_OBJECT
PUSH_OBJECT
PULL_OBJECT
ROTATE_OBJECT
SCALE_OBJECT
POINT_AT_OBJECT
```

Эти события требуют информации не только о человеке, но и об объекте.

---

# 46. Composite gestures

Composite gesture = комбинация нескольких условий.

Пример:

```text
right arm forward
+
right hand fist
+
high wrist velocity
=
RIGHT_PUNCH
```

```text
both arms above head
+
both palms open
=
HANDS_UP_POSE
```

```text
index extended
+
other fingers folded
=
POINT
```

---

# 47. Gesture lifecycle

Для жестов с продолжительностью использовать единый lifecycle:

```text
GESTURE_STARTED
GESTURE_ACTIVE
GESTURE_UPDATED
GESTURE_ENDED
GESTURE_CANCELLED
```

Например pinch:

```text
PINCH_STARTED
PINCH_ACTIVE
PINCH_MOVED
PINCH_RELEASED
```

---

# 48. Cooldown

Одно физическое действие не должно генерировать десятки одинаковых событий.

Пример настроек:

```text
PUNCH_COOLDOWN_MS = 300
SWIPE_COOLDOWN_MS = 250
JUMP_COOLDOWN_MS = 500
```

Значения должны лежать в config.

---

# 49. Debounce

Статические жесты не должны переключаться из-за шума landmarks.

Использовать правило:

```text
condition must remain valid
for N frames
or N milliseconds
```

до подтверждения нового состояния.

---

# 50. Hysteresis

Для пороговых условий желательно использовать разные пороги входа и выхода.

Например:

```text
PINCH_ENTER_DISTANCE = 0.04
PINCH_EXIT_DISTANCE = 0.06
```

---

# 51. Normalized distances

Не использовать абсолютные pixel-distance как единственный критерий.

Для кисти расстояния лучше нормализовать относительно размера руки, например:

```text
distance(wrist, middle_mcp)
```

Для тела — относительно:

```text
shoulder width
```

или:

```text
torso length
```

---

# 52. Angles

Универсальная функция:

```python
angle(a, b, c)
```

возвращает угол ABC.

Примеры:

```text
shoulder → elbow → wrist
```

для elbow angle.

```text
hip → knee → ankle
```

для knee angle.

```text
shoulder → hip → knee
```

для hip angle.

---

# 53. Velocity

Для динамических действий нужна скорость:

```text
velocity =
(current_position - previous_position) / dt
```

Использовать для:

```text
PUNCH
SWIPE
THROW
WAVE
KICK
```

---

# 54. Acceleration

```text
acceleration =
(current_velocity - previous_velocity) / dt
```

Использовать только если velocity недостаточно.

Не усложнять алгоритм без необходимости.

---

# 55. Direction

Для движения вычислять normalized vector:

```text
dx
dy
dz
```

и классифицировать:

```text
LEFT
RIGHT
UP
DOWN
FORWARD
BACKWARD
```

---

# 56. Gesture history

Event Engine должен хранить ограниченную историю, например:

```text
500–1500 ms
```

История может включать:

```text
timestamp
landmarks
angles
distances
velocity
states
```

Не хранить бесконечную историю.

---

# 57. State machine

Сложные действия реализовывать через state machine.

Squat:

```text
IDLE
↓
GOING_DOWN
↓
BOTTOM
↓
GOING_UP
↓
COMPLETE
```

Punch:

```text
IDLE
↓
WINDUP
↓
EXTENDING
↓
HIT
↓
RECOVERY
↓
IDLE
```

Jump:

```text
GROUND
↓
TAKEOFF
↓
AIRBORNE
↓
LANDING
↓
GROUND
```

---

# 58. Confidence

Каждое распознанное событие может иметь:

```python
confidence: float
```

Диапазон:

```text
0.0 .. 1.0
```

Confidence события не обязан совпадать с MediaPipe confidence.

Он может вычисляться из:

```text
landmark visibility
+
rule strength
+
temporal consistency
```

---

# 59. Suggested Event model

```python
from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    PERSON_DETECTED = auto()
    PERSON_LOST = auto()

    LEFT_ARM_RAISED = auto()
    RIGHT_ARM_RAISED = auto()

    LEFT_PINCH_STARTED = auto()
    LEFT_PINCH_RELEASED = auto()

    RIGHT_PINCH_STARTED = auto()
    RIGHT_PINCH_RELEASED = auto()

    LEFT_PUNCH = auto()
    RIGHT_PUNCH = auto()

    CLAP = auto()

    JUMP = auto()
    LAND = auto()

    SQUAT_REP = auto()


@dataclass(frozen=True)
class GestureEvent:
    type: EventType
    timestamp_ms: int
    confidence: float
```

Это только пример структуры. Не ограничивать полный список событий данным примером.

---

# 60. Suggested FrameState

```python
@dataclass(frozen=True)
class FrameState:
    timestamp_ms: int

    left_hand_visible: bool
    right_hand_visible: bool

    left_arm_up: bool
    right_arm_up: bool

    left_hand_shape: str | None
    right_hand_shape: str | None

    left_pinch: bool
    right_pinch: bool
```

Реальная структура может быть более универсальной.

---

# 61. Категории EventType

```text
TRACKING
HEAD
TORSO
LEFT_ARM
RIGHT_ARM
LEFT_HAND
RIGHT_HAND
FINGER
BODY_POSE
MOVEMENT
FITNESS
COMBAT
UI
OBJECT_INTERACTION
COMPOSITE
```

---

# 62. MVP набор событий

Не реализовывать весь каталог сразу.

Первый полезный набор:

```text
PERSON_DETECTED
PERSON_LOST
LEFT_HAND_DETECTED
RIGHT_HAND_DETECTED
LEFT_ARM_RAISED
RIGHT_ARM_RAISED
BOTH_ARMS_RAISED
LEFT_OPEN_PALM
RIGHT_OPEN_PALM
LEFT_FIST
RIGHT_FIST
LEFT_POINT
RIGHT_POINT
LEFT_PINCH_STARTED
LEFT_PINCH_RELEASED
RIGHT_PINCH_STARTED
RIGHT_PINCH_RELEASED
LEFT_SWIPE_LEFT
LEFT_SWIPE_RIGHT
RIGHT_SWIPE_LEFT
RIGHT_SWIPE_RIGHT
LEFT_WAVE
RIGHT_WAVE
LEFT_PUNCH
RIGHT_PUNCH
CLAP
LEAN_LEFT
LEAN_RIGHT
SQUAT_STARTED
SQUAT_REP
JUMP
LAND
```

---

# 63. Второй этап

После стабильного MVP:

```text
SWIPE_UP
SWIPE_DOWN
GRAB
PUSH
PULL
THROW
KICK
STEP
JUMPING_JACK_REP
LUNGE
TWO_HAND_SCALE
TWO_HAND_ROTATE
```

---

# 64. Не реализовывать всё одним большим if

Рекомендуемая архитектура:

```text
geometry.py
    angles
    distances
    vectors

hand_state.py
    fingers
    palm
    fist
    pinch

body_state.py
    arms
    torso
    legs
    static poses

motion_history.py
    velocity
    trajectories

gesture_detector.py
    swipe
    wave
    punch
    clap

action_detector.py
    squat
    jump
    lunge
    jumping jack

events.py
    EventType
    GestureEvent

event_engine.py
    transitions
    cooldown
    debounce
```

---

# 65. Separation from MediaPipe

Gesture logic не должна зависеть напрямую от MediaPipe classes.

Плохо:

```python
def detect_punch(mp_pose_result):
    ...
```

Лучше:

```python
def detect_punch(frame_state, history):
    ...
```

MediaPipe должен преобразовываться во внутренний контракт раньше.

---

# 66. Pose + Hand synchronization

Pose и Hand Landmarker могут выдавать результаты в разное время.

Для объединения использовать timestamp.

Не объединять результаты, если временная разница слишком большая.

Например:

```text
MAX_SYNC_DELTA_MS
```

должен находиться в config.

---

# 67. Hand association

Hand Landmarker должен быть сопоставлен с:

```text
LEFT
RIGHT
```

Не полагаться только на положение руки на экране.

После зеркалирования UI визуальное положение не должно менять анатомическую сторону.

---

# 68. Lost tracking

Если рука потеряна:

```text
HAND_LOST
```

активные состояния должны корректно завершаться.

Например:

```text
PINCH_ACTIVE
↓
hand lost
↓
PINCH_CANCELLED
```

а не:

```text
PINCH_RELEASED
```

Потеря трекинга и реальное физическое отпускание — разные события.

---

# 69. Event priority

Если несколько событий конфликтуют, можно использовать приоритет.

Например `RIGHT_PUNCH` может иметь больший приоритет, чем `RIGHT_ARM_EXTENDED`.

Но оба события могут существовать одновременно, если это полезно приложению.

---

# 70. Semantic mapping layer

Физические жесты и действия приложения должны быть разделены.

Пример:

```text
RIGHT_SWIPE_RIGHT
↓
NEXT
```

```python
gesture_map = {
    EventType.RIGHT_SWIPE_RIGHT: Action.NEXT,
    EventType.RIGHT_PINCH_STARTED: Action.SELECT,
}
```

Это позволит менять управление без переписывания Gesture Engine.

---

# 71. Общий каталог групп событий

```text
TRACKING
HEAD
SHOULDER
ARM
TORSO
BODY_POSE
LEG
FINGER
HAND_SHAPE
PINCH
GRAB
POINT
SWIPE
WAVE
PUNCH
PUSH
PULL
CIRCLE
CLAP
THROW
CATCH
JUMP
SQUAT
LUNGE
STEP
WALK
RUN
JUMPING_JACK
DODGE
KICK
BLOCK
FITNESS
COMBAT
POINTER
DRAG
TWO_HAND_UI
NAVIGATION
OBJECT_INTERACTION
COMPOSITE
```

---

# 72. Главный принцип

MediaPipe отвечает на вопрос:

```text
Где находятся точки тела и кистей?
```

Geometry layer отвечает:

```text
Какие между ними углы, расстояния и направления?
```

Frame State отвечает:

```text
В каком положении сейчас находится человек?
```

Temporal Engine отвечает:

```text
Как это положение изменялось во времени?
```

Gesture Engine отвечает:

```text
Какой жест или действие произошло?
```

Application отвечает:

```text
Что сделать в игре или программе после этого события?
```

Не смешивать эти уровни.
