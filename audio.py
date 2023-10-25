import moviepy.editor as mpe


video = "mandelbrot_animation.mp4"
audio = "audio.mp3"

video_clip = mpe.VideoFileClip(video)
audio_clip = mpe.AudioFileClip(audio)
result = video_clip.set_audio(audio_clip)
result.write_videofile("mandelbrot_animation_with_audio.mp4", fps=24, codec='mpeg4')
