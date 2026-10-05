package io.github.shantrapa.motion

/** Port of motion/contract.py. Coordinates are normalized to the upright, unmirrored camera frame. */
data class Landmark(
    val x: Double,          // 0..1 across frame width, may go outside
    val y: Double,          // 0..1 across frame height
    val z: Double,          // depth relative to hips, smaller = closer to camera
    val visibility: Double, // 0..1
)

data class PoseFrame(
    val timestampMs: Long,             // capture time of the frame
    val landmarks: List<Landmark>?,    // 33 points, or null if no person
)

const val NUM_LANDMARKS = 33

// Odd index = person's left side, even = right.
const val NOSE = 0
const val LEFT_SHOULDER = 11
const val RIGHT_SHOULDER = 12
const val LEFT_ELBOW = 13
const val RIGHT_ELBOW = 14
const val LEFT_WRIST = 15
const val RIGHT_WRIST = 16
const val LEFT_PINKY = 17
const val RIGHT_PINKY = 18
const val LEFT_INDEX = 19
const val RIGHT_INDEX = 20
const val LEFT_HIP = 23
const val RIGHT_HIP = 24
const val LEFT_KNEE = 25
const val RIGHT_KNEE = 26
const val LEFT_ANKLE = 27
const val RIGHT_ANKLE = 28

val SKELETON: List<Pair<Int, Int>> = listOf(
    // face
    0 to 1, 1 to 2, 2 to 3, 3 to 7, 0 to 4, 4 to 5, 5 to 6, 6 to 8, 9 to 10,
    // torso
    11 to 12, 11 to 23, 12 to 24, 23 to 24,
    // arms and hands
    11 to 13, 13 to 15, 15 to 17, 15 to 19, 15 to 21, 17 to 19,
    12 to 14, 14 to 16, 16 to 18, 16 to 20, 16 to 22, 18 to 20,
    // legs and feet
    23 to 25, 25 to 27, 27 to 29, 29 to 31, 27 to 31,
    24 to 26, 26 to 28, 28 to 30, 30 to 32, 28 to 32,
)
