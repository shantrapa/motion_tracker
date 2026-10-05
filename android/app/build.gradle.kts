import java.net.URI

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "io.github.shantrapa.motion"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.shantrapa.motion"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")  // phones + emulator; drops ~40 MB of 32-bit MediaPipe libs
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    androidResources {
        noCompress += "task"  // MediaPipe memory-maps models from assets
    }
}

// Models are not committed; fetch them into assets before every build (no-op when present).
val modelsDir = layout.projectDirectory.dir("src/main/assets")
val modelUrls = mapOf(
    "pose_landmarker_full.task" to
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
)
val downloadModels by tasks.registering {
    outputs.files(modelUrls.keys.map { modelsDir.file(it) })
    doLast {
        modelUrls.forEach { (name, url) ->
            val dest = modelsDir.file(name).asFile
            if (dest.exists()) return@forEach
            dest.parentFile.mkdirs()
            val tmp = File(dest.path + ".part")  // no half-written model if the download dies
            logger.lifecycle("downloading $url")
            URI(url).toURL().openStream().use { input -> tmp.outputStream().use { input.copyTo(it) } }
            check(tmp.renameTo(dest)) { "cannot move $tmp to $dest" }
        }
    }
}
tasks.named("preBuild") { dependsOn(downloadModels) }

dependencies {
    val camerax = "1.5.3"  // 1.6 needs compileSdk 36
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("com.google.mediapipe:tasks-vision:1.0.0")

    testImplementation("junit:junit:4.13.2")
}
