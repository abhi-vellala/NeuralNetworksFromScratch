import ffmpeg
(
    ffmpeg
    .input('UnetFromScratch/unet_explain_figs/*.png', pattern_type='glob', framerate=10)
    .output('movie.mp4')
    .run()
)