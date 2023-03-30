import paddle

mode = 'dev'
tess_dataset = paddle.audio.datasets.TESS(mode=mode,
                                        feat_type='raw')
for idx in range(5):
    audio, label = tess_dataset[idx]
    # do something with audio, label
    print(audio.shape, label)
    # [audio_data_length] , label_id

tess_dataset = paddle.audio.datasets.TESS(mode=mode,
                                        feat_type='mfcc',
                                        n_mfcc=40)
for idx in range(5):
    audio, label = tess_dataset[idx]
    # do something with mfcc feature, label
    print(audio.shape, label)
    # [feature_dim, num_frames] , label_id