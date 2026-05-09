import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame extraction and processing"""
    
    def __init__(self):
        pass
    
    def extract_frames(self, video_path, sample_rate=10):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame (default: 10)
            
        Returns:
            List of frames (numpy arrays)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Video info: {total_frames} frames, {fps} FPS")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Sample every Nth frame
                if frame_count % sample_rate == 0:
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from {total_frames} total")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def extract_key_frames(self, video_path, num_frames=5):
        """
        Extract evenly spaced key frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frames (numpy arrays)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} key frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting key frames: {str(e)}")
            raise
    
    def get_video_info(self, video_path):
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            info = {
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise


# For testing
if __name__ == "__main__":
    processor = VideoProcessor()
    print("VideoProcessor initialized")
