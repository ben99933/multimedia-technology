import cv2
import moviepy.editor as mpe

video = "mandelbrot_animation.mp4"
audio = "audio.mp3"

video_clip = mpe.VideoFileClip(video)
audio_clip = mpe.AudioFileClip(audio)
result = video_clip.set_audio(audio_clip)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
(width, height) = (1024, 1024)
fps = 24
result.write_videofile("mandelbrot_animation_with_audio.mp4", fps=fps, codec="libx264", audio_codec="aac")
