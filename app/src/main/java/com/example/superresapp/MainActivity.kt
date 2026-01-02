package com.example.superresapp

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import org.opencv.features2d.ORB
import org.opencv.features2d.DescriptorMatcher
import org.opencv.calib3d.Calib3d
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfPoint2f
import org.opencv.core.DMatch

class MainActivity : AppCompatActivity() {

    // --- UI Components ---
    private lateinit var mainLayout: View
    private lateinit var tvStatus: TextView
    private lateinit var ivResult: ImageView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnSave: Button
    private lateinit var spinnerUpscale: Spinner
    private lateinit var seekBarBurst: SeekBar
    private lateinit var tvBurstLabel: TextView
    private lateinit var seekBarSpeed: SeekBar
    private lateinit var tvSpeedLabel: TextView
    private lateinit var cameraContainer: FrameLayout
    private lateinit var viewFinder: PreviewView
    private lateinit var btnCapture: Button
    private lateinit var tvCameraStatus: TextView

    // --- Data ---
    private var resultBitmap: Bitmap? = null
    private var burstCount = 4
    private var upscaleFactor = 2

    // --- CameraX ---
    private var imageAnalysis: ImageAnalysis? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val isBursting = AtomicBoolean(false)
    private val capturedBitmaps = ArrayList<Bitmap>()
    private var framesTaken = 0

    // --- Permissions ---
    private val requestStorageLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            resultBitmap?.let { saveImageToGallery(it) }
        } else {
            Toast.makeText(this, "Permission required to save!", Toast.LENGTH_SHORT).show()
        }
    }

    private val requestCameraLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
    }

    private val pickImagesLauncher = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.size >= 2) processImages(uris = uris)
        else Toast.makeText(this, "Select at least 2 images", Toast.LENGTH_SHORT).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (OpenCVLoader.initDebug()) Log.d("OpenCV", "Success")

        // Bind UI
        mainLayout = findViewById(R.id.mainLayout)
        tvStatus = findViewById(R.id.tvStatus)
        ivResult = findViewById(R.id.ivResult)
        progressBar = findViewById(R.id.progressBar)
        btnSave = findViewById(R.id.btnSave)
        spinnerUpscale = findViewById(R.id.spinnerUpscale)
        seekBarBurst = findViewById(R.id.seekBarBurst)
        tvBurstLabel = findViewById(R.id.tvBurstLabel)
        seekBarSpeed = findViewById(R.id.seekBarSpeed)
        tvSpeedLabel = findViewById(R.id.tvSpeedLabel)
        cameraContainer = findViewById(R.id.cameraContainer)
        viewFinder = findViewById(R.id.viewFinder)
        btnCapture = findViewById(R.id.btnCapture)
        tvCameraStatus = findViewById(R.id.tvCameraStatus)

        setupListeners()
    }

    private fun setupListeners() {
        findViewById<Button>(R.id.btnGallery).setOnClickListener { pickImagesLauncher.launch("image/*") }

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
            else requestCameraLauncher.launch(Manifest.permission.CAMERA)
        }

        findViewById<Button>(R.id.btnCloseCamera).setOnClickListener { closeCameraUI() }

        btnSave.setOnClickListener {
            if (resultBitmap != null) {
                // Android 9 and older needs explicit permission request
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                    requestStorageLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                } else {
                    saveImageToGallery(resultBitmap!!)
                }
            }
        }

        // Burst Slider
        seekBarBurst.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(p0: SeekBar?, p: Int, p2: Boolean) { burstCount = p + 2; tvBurstLabel.text = "Shots: $burstCount" }
            override fun onStartTrackingTouch(p0: SeekBar?) {}
            override fun onStopTrackingTouch(p0: SeekBar?) {}
        })

        // Speed Slider
        seekBarSpeed.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(p0: SeekBar?, p: Int, p2: Boolean) {
                tvSpeedLabel.text = if (p == 0) "Delay: 0ms (Max)" else "Delay: ${p}ms"
            }
            override fun onStartTrackingTouch(p0: SeekBar?) {}
            override fun onStopTrackingTouch(p0: SeekBar?) {}
        })

        // Capture Button
        btnCapture.setOnClickListener {
            capturedBitmaps.clear()
            framesTaken = 0
            isBursting.set(true)
            btnCapture.isEnabled = false
            tvCameraStatus.visibility = View.VISIBLE
            tvCameraStatus.text = "0 / $burstCount"
        }
    }

    // ==========================================
    // FAST CAMERA (STREAM ANALYSIS)
    // ==========================================

    private fun startCamera() {
        mainLayout.visibility = View.GONE
        cameraContainer.visibility = View.VISIBLE

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            // Stream Analysis for Speed
            imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1920, 1080)) // Full HD is safer for memory
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis?.setAnalyzer(cameraExecutor) { imageProxy ->
                if (isBursting.get()) {
                    val bitmap = imageProxy.toBitmap()
                    val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)

                    synchronized(capturedBitmaps) {
                        capturedBitmaps.add(rotatedBitmap)
                        framesTaken++
                    }

                    runOnUiThread { tvCameraStatus.text = "$framesTaken / $burstCount" }

                    if (framesTaken >= burstCount) {
                        isBursting.set(false)
                        runOnUiThread {
                            tvCameraStatus.text = "Processing..."
                            closeCameraUI()
                            btnCapture.isEnabled = true
                            processImages(bitmaps = ArrayList(capturedBitmaps))
                        }
                    }
                    val delay = seekBarSpeed.progress.toLong()
                    if (delay > 0) Thread.sleep(delay)
                }
                imageProxy.close()
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
            } catch (e: Exception) {
                Log.e("Camera", "Bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun closeCameraUI() {
        cameraContainer.visibility = View.GONE
        mainLayout.visibility = View.VISIBLE
    }

    // ==========================================
    // PROCESSING
    // ==========================================

    private fun processImages(uris: List<Uri>? = null, bitmaps: List<Bitmap>? = null) {
        val selectedStr = spinnerUpscale.selectedItem.toString()
        upscaleFactor = selectedStr.replace("x", "").toIntOrNull() ?: 2

        progressBar.visibility = View.VISIBLE
        btnSave.isEnabled = false
        tvStatus.text = "Aligning & Processing..."
        ivResult.setImageBitmap(null)

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val images = ArrayList<Mat>()

                // 1. Load Images
                if (uris != null) {
                    for (uri in uris) {
                        val bmp = loadBitmapFromUri(uri) ?: continue
                        val mat = Mat()
                        Utils.bitmapToMat(bmp, mat)
                        if (mat.channels() == 4) Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR)
                        images.add(mat)
                    }
                } else if (bitmaps != null) {
                    for (bmp in bitmaps) {
                        val mat = Mat()
                        Utils.bitmapToMat(bmp, mat)
                        if (mat.channels() == 4) Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR)
                        images.add(mat)
                    }
                }

                if (images.size < 2) throw Exception("Need 2+ images")

                // 2. Align All Images to the First One (Reference)
                val refImg = images[0]
                val alignedImages = ArrayList<Mat>()
                alignedImages.add(refImg) // First one is perfect by definition

                for (i in 1 until images.size) {
                    val targetImg = images[i]

                    // --- NEW ALIGNMENT LOGIC ---
                    // Using ORB to fix Rotation and Shift
                    val aligned = alignImageUsingORB(refImg, targetImg)
                    alignedImages.add(aligned)

                    // Note: We don't release targetImg yet as it might be returned if align fails
                }

                // 3. Fusion & Upscaling
                val h = refImg.rows()
                val w = refImg.cols()
                val highW = w * upscaleFactor
                val highH = h * upscaleFactor

                // Accumulator (Float 32)
                val accumulator = Mat.zeros(Size(highW.toDouble(), highH.toDouble()), CvType.CV_32FC3)

                for (img in alignedImages) {
                    val upscaled = Mat()
                    // Lanczos4 is the best resizing for upscaling
                    Imgproc.resize(img, upscaled, Size(highW.toDouble(), highH.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)

                    val upscaledFloat = Mat()
                    upscaled.convertTo(upscaledFloat, CvType.CV_32FC3)

                    Core.add(accumulator, upscaledFloat, accumulator)

                    upscaled.release()
                    upscaledFloat.release()
                }

                // Average
                val resultFloat = Mat()
                Core.multiply(accumulator, Scalar.all(1.0 / alignedImages.size), resultFloat)

                // Convert to 8-bit Image
                val finalMat = Mat()
                resultFloat.convertTo(finalMat, CvType.CV_8UC3)
                Imgproc.cvtColor(finalMat, finalMat, Imgproc.COLOR_BGR2RGBA)

                // 4. Output
                val highResBitmap = Bitmap.createBitmap(highW, highH, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(finalMat, highResBitmap)

                // Preview (1/4 size)
                val previewBitmap = Bitmap.createScaledBitmap(highResBitmap, highW / 4, highH / 4, true)

                // Cleanup Mats to free RAM
                accumulator.release(); resultFloat.release(); finalMat.release()
                for(img in images) img.release()
                // Note: alignedImages contains references to 'images' or new mats.
                // To be safe, we rely on 'images' cleanup mostly, but ideal to clear list.

                withContext(Dispatchers.Main) {
                    resultBitmap = highResBitmap
                    ivResult.setImageBitmap(previewBitmap)
                    tvStatus.text = "Success! ${highW}x${highH}"
                    progressBar.visibility = View.GONE
                    btnSave.isEnabled = true
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Error: ${e.message}"
                    progressBar.visibility = View.GONE
                }
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
    }

    // ==========================================
    // SAVING LOGIC (ANDROID 9 FIXED)
    // ==========================================

    private fun saveImageToGallery(bitmap: Bitmap) {
        val filename = "SuperRes_${System.currentTimeMillis()}.jpg"

        // --- METHOD A: Android 10+ (Q and newer) ---
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val cv = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv) ?: return

            try {
                contentResolver.openOutputStream(uri)?.use {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it)
                }
                cv.clear()
                cv.put(MediaStore.MediaColumns.IS_PENDING, 0)
                contentResolver.update(uri, cv, null, null)
                Toast.makeText(this, "Saved to Gallery!", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Save Failed: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }

        // --- METHOD B: Android 9 (Pie) and older ---
        else {
            try {
                // 1. Get the directory and CREATE it if missing
                val imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                if (!imagesDir.exists()) {
                    imagesDir.mkdirs()
                }

                // 2. Create the file
                val imageFile = File(imagesDir, filename)
                val fos = FileOutputStream(imageFile)

                // 3. Write the data
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
                fos.flush()
                fos.close()

                // 4. CRITICAL: Scan the file so Gallery sees it
                android.media.MediaScannerConnection.scanFile(
                    this,
                    arrayOf(imageFile.absolutePath),
                    arrayOf("image/jpeg")
                ) { path, uri ->
                    // Run on main thread to show Toast
                    runOnUiThread {
                        Toast.makeText(this, "Saved to Gallery!", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                Toast.makeText(this, "Legacy Save Failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
    // --- NEW: Advanced Alignment using ORB Features ---
    private fun alignImageUsingORB(refMat: Mat, targetMat: Mat): Mat {
        // 1. Detect Features (Keypoints)
        val orb = ORB.create(500) // Find top 500 features
        val keypointsRef = MatOfKeyPoint()
        val descriptorsRef = Mat()
        orb.detectAndCompute(refMat, Mat(), keypointsRef, descriptorsRef)

        val keypointsTarget = MatOfKeyPoint()
        val descriptorsTarget = Mat()
        orb.detectAndCompute(targetMat, Mat(), keypointsTarget, descriptorsTarget)

        if (descriptorsRef.empty() || descriptorsTarget.empty()) {
            return targetMat // Alignment failed, return original (or skip)
        }

        // 2. Match Features
        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        val matches = MatOfDMatch()
        matcher.match(descriptorsTarget, descriptorsRef, matches)

        // 3. Filter for "Good" Matches (Remove noise)
        val matchList = matches.toList()
        matchList.sortBy { it.distance } // Sort by best match

        // Keep top 15% of matches or at least 10 points
        val goodMatchesCount = (matchList.size * 0.15).toInt().coerceAtLeast(10)
        if (matchList.size < goodMatchesCount) return targetMat // Not enough matches

        val goodMatches = matchList.take(goodMatchesCount)

        // 4. Extract coordinates of matching points
        val ptsRef = ArrayList<Point>()
        val ptsTarget = ArrayList<Point>()

        val keypointsRefList = keypointsRef.toList()
        val keypointsTargetList = keypointsTarget.toList()

        for (match in goodMatches) {
            ptsTarget.add(keypointsTargetList[match.queryIdx].pt)
            ptsRef.add(keypointsRefList[match.trainIdx].pt)
        }

        val matPtsRef = MatOfPoint2f(*ptsRef.toTypedArray())
        val matPtsTarget = MatOfPoint2f(*ptsTarget.toTypedArray())

        // 5. Compute Homography (The Magic Matrix that handles Rotation & Tilt)
        // RANSAC ignores outlier points that might be wrong
        val homography = Calib3d.findHomography(matPtsTarget, matPtsRef, Calib3d.RANSAC, 5.0)

        if (homography.empty()) return targetMat

        // 6. Warp the Target image to match Reference
        val aligned = Mat()
        Imgproc.warpPerspective(targetMat, aligned, homography, refMat.size())

        return aligned
    }
}