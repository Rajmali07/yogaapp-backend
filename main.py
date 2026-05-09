from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import asyncio
import uuid
from pathlib import Path
import logging
import base64
import cv2
from typing import List

from model_handler import YogaModelHandler
from video_processor import VideoProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Yoga Pose Correction API")

# Keep Render requests smaller and faster.
# Returning base64 images for every frame can create very large responses.
MAX_FRAMES_TO_ANALYZE = int(os.environ.get("MAX_FRAMES_TO_ANALYZE", "12"))
INCLUDE_FRAME_IMAGES = os.environ.get("INCLUDE_FRAME_IMAGES", "true").lower() == "true"
IS_RENDER = bool(os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID"))
FRAME_SAMPLE_RATE = int(os.environ.get("FRAME_SAMPLE_RATE", "20" if IS_RENDER else "10"))
ASYNC_ANALYSIS = os.environ.get("ASYNC_ANALYSIS", "").lower() == "true" or IS_RENDER
FRAME_IMAGE_MAX_WIDTH = int(os.environ.get("FRAME_IMAGE_MAX_WIDTH", "240" if IS_RENDER else "320"))
FRAME_IMAGE_JPEG_QUALITY = int(os.environ.get("FRAME_IMAGE_JPEG_QUALITY", "45" if IS_RENDER else "70"))

def _csv_env(name: str) -> List[str]:
    raw_value = os.environ.get(name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _frame_to_data_url(frame) -> str:
    """Convert a frame to a compressed JPEG data URL for UI previews."""
    height, width = frame.shape[:2]
    if width > FRAME_IMAGE_MAX_WIDTH:
        scale = FRAME_IMAGE_MAX_WIDTH / float(width)
        frame = cv2.resize(frame, (FRAME_IMAGE_MAX_WIDTH, int(height * scale)))

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_IMAGE_JPEG_QUALITY]
    ok, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode frame preview")
    frame_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{frame_base64}"


# Enable CORS for local development and hosted frontends.
# You can override this with ALLOWED_ORIGINS or FRONTEND_ORIGIN on Render.
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:4200",
    "http://localhost:5000",
    "http://localhost:8000",
    "http://localhost:8091",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8091",
]

allowed_origins.extend(_csv_env("ALLOWED_ORIGINS"))

frontend_origin = os.environ.get("FRONTEND_ORIGIN")
if frontend_origin:
    allowed_origins.append(frontend_origin)

allowed_origin_regex = os.environ.get(
    "ALLOWED_ORIGIN_REGEX",
    r"https://.*\.netlify\.app|https://.*\.onrender\.com|http://(localhost|127\.0\.0\.1)(:\d+)?",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize video processor
video_processor = VideoProcessor()

# Model will be loaded lazily.
# Keep the default path inside this repo; override with MODEL_PATH if needed.
def _resolve_saved_model_path() -> str:
    raw_path = os.environ.get("MODEL_PATH")
    if raw_path:
        return raw_path

    here = Path(__file__).resolve().parent
    candidates = [
        here / "model_prep" / "yoga_savedmodel",
        here / "model_prep" / "yoga_savedmodel" / "yoga_savedmodel",
        here.parent / "model_prep" / "yoga_savedmodel",
        here.parent / "model_prep" / "yoga_savedmodel" / "yoga_savedmodel",
    ]

    for candidate in candidates:
        if (candidate / "saved_model.pb").exists() or (candidate / "saved_model.pbtxt").exists():
            logger.info("Resolved SavedModel path to %s", candidate)
            return str(candidate)

    logger.warning(
        "Could not auto-resolve SavedModel path. Falling back to %s",
        candidates[0],
    )
    return str(candidates[0])


SAVED_MODEL_PATH = _resolve_saved_model_path()
model_handler = None

# Create temp directory for uploads
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory job store for async analysis on Render.
# This is enough for a single-instance deployment and keeps request/response times short.
analysis_jobs = {}


def _get_model_handler() -> YogaModelHandler:
    """Create and lazily load the TensorFlow model only when needed."""
    global model_handler

    if model_handler is None:
        model_handler = YogaModelHandler(SAVED_MODEL_PATH)
        logger.info("Model handler initialized")

    if model_handler.model is None:
        logger.info("Loading TensorFlow model lazily...")
        model_handler.load_model()
        logger.info("TensorFlow model loaded successfully")

    return model_handler


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Server starting up...")
    if not os.path.exists(SAVED_MODEL_PATH):
        logger.warning(
            "TensorFlow model directory not found at %s. "
            "Set MODEL_PATH or add model_prep/yoga_savedmodel to this repo.",
            SAVED_MODEL_PATH,
        )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Yoga Pose Correction API is running"}


@app.post("/analyze-pose")
async def analyze_pose(
    video: UploadFile = File(...),
    expected_pose: str = Form(None)
):
    """
    Analyze yoga pose from uploaded video
    
    Args:
        video: Video file (mp4, avi, mov)
        expected_pose: The expected yoga asana name
        
    Returns:
        JSON with pose analysis results
    """
    video_path = None
    cleanup_in_finally = True
    
    try:
        # In async/Render mode, do not block the request on model loading.
        # The background job will load the model if needed.
        if not ASYNC_ANALYSIS:
            _get_model_handler()

        # Validate file type (check content type or file extension)
        valid_video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        is_video_content = video.content_type and video.content_type.startswith('video/')
        is_video_extension = any(video.filename.lower().endswith(ext) for ext in valid_video_extensions)
        
        if not is_video_content and not is_video_extension:
            logger.error(f"Invalid file type: {video.content_type}, filename: {video.filename}")
            raise HTTPException(400, "File must be a video (mp4, avi, mov, mkv, or webm)")
        
        # Save uploaded video temporarily
        video_path = UPLOAD_DIR / f"temp_{video.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        if ASYNC_ANALYSIS:
            job_id = str(uuid.uuid4())
            analysis_jobs[job_id] = {
                "status": "queued",
                "message": "Analysis queued",
                "video_name": video.filename,
            }
            cleanup_in_finally = False
            asyncio.create_task(_run_analysis_job(job_id, str(video_path), video.filename, expected_pose))
            return JSONResponse(
                status_code=202,
                content={
                    "status": "queued",
                    "job_id": job_id,
                    "message": "Analysis started in the background",
                },
            )

        overall_result = await asyncio.to_thread(
            _analyze_video_file,
            str(video_path),
            video.filename,
            expected_pose,
        )
        return JSONResponse(content=overall_result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing video")
        raise HTTPException(500, f"Error processing video: {str(e)}")
    finally:
        if cleanup_in_finally and video_path and video_path.exists():
            os.remove(video_path)


async def _run_analysis_job(job_id: str, video_path: str, video_name: str, expected_pose: str):
    analysis_jobs[job_id] = {
        "status": "processing",
        "message": "Analysis in progress",
        "video_name": video_name,
    }

    try:
        result = await asyncio.to_thread(_analyze_video_file, video_path, video_name, expected_pose)
        analysis_jobs[job_id] = {
            "status": "completed",
            "message": "Analysis complete",
            "video_name": video_name,
            "result": result,
        }
    except Exception as e:
        logger.exception("Background analysis failed")
        analysis_jobs[job_id] = {
            "status": "failed",
            "message": str(e),
            "video_name": video_name,
        }
    finally:
        if Path(video_path).exists():
            os.remove(video_path)


def _analyze_video_file(video_path: str, video_name: str, expected_pose: str):
    logger.info("Processing video: %s", video_name)
    _get_model_handler()

    # Extract frames from video
    frames = video_processor.extract_frames(video_path, sample_rate=FRAME_SAMPLE_RATE)
    if not frames:
        raise HTTPException(400, "No frames could be extracted from the uploaded video")

    # Cap the number of frames we send through the model and back to the client.
    # This keeps hosted deployments from timing out or exhausting memory on large videos.
    render_frame_limit = int(os.environ.get("MAX_FRAMES_TO_ANALYZE_RENDER", "6" if IS_RENDER else str(MAX_FRAMES_TO_ANALYZE)))
    if len(frames) > render_frame_limit:
        frames = frames[:render_frame_limit]

    logger.info("Extracted %s frames", len(frames))

    results = []
    for idx, frame in enumerate(frames):
        prediction = model_handler.predict(frame)

        frame_result = {
            "frame_number": idx,
            "pose_detected": prediction["pose_class"],
            "confidence": prediction["confidence"],
            "is_correct": prediction["is_correct"],
            "feedback": prediction["feedback"],
        }

        if INCLUDE_FRAME_IMAGES:
            frame_result["image"] = _frame_to_data_url(frame)

        results.append(frame_result)

    if expected_pose:
        correct_count = sum(1 for r in results if r["pose_detected"] == expected_pose and r["confidence"] > 0.7)
    else:
        correct_count = sum(1 for r in results if r["is_correct"])

    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    return {
        "video_name": video_name,
        "expected_pose": expected_pose,
        "total_frames_analyzed": len(frames),
        "correct_frames": correct_count,
        "incorrect_frames": len(frames) - correct_count,
        "accuracy_percentage": round((correct_count / len(frames)) * 100, 2),
        "average_confidence": round(avg_confidence, 2),
        "frame_results": results,
        "overall_feedback": _generate_overall_feedback(results, expected_pose),
    }


@app.get("/analysis-status/{job_id}")
async def analysis_status(job_id: str):
    job = analysis_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Analysis job not found")
    return JSONResponse(content=job)


@app.post("/analyze-webcam-frame")
async def analyze_webcam_frame(frame: UploadFile = File(...)):
    """
    Analyze single frame from webcam
    
    Args:
        frame: Image file (jpg, png)
        
    Returns:
        JSON with pose analysis result
    """
    try:
        _get_model_handler()
        
        import cv2
        import numpy as np
        
        # Read image file
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze frame
        prediction = model_handler.predict(img)
        
        return JSONResponse(content=prediction)
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise HTTPException(500, f"Error processing frame: {str(e)}")


def _generate_overall_feedback(results, expected_pose=None):
    """Generate human-readable overall feedback"""
    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = (correct_count / len(results)) * 100
    
    feedback = ""
    if expected_pose:
        feedback = f"Expected: {expected_pose}. "
    
    if accuracy >= 90:
        feedback += "Excellent! Your form is nearly perfect. Keep it up!"
    elif accuracy >= 70:
        feedback += "Good job! Minor adjustments needed in some frames."
    elif accuracy >= 50:
        feedback += "Decent attempt. Focus on maintaining proper form throughout."
    else:
        feedback += "Needs improvement. Review the pose guidelines and try again."
    
    return feedback


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
