import paddle

mode = 'dev'
esc50_dataset = paddle.audio.datasets.ESC50(mode=mode,
                                        feat_type='raw')
for idx in range(5):
    audio, label = esc50_dataset[idx]
    # do something with audio, label
    print(audio.shape, label)
    # [audio_data_length] , label_id

esc50_dataset = paddle.audio.datasets.ESC50(mode=mode,
                                        feat_type='mfcc',
                                        n_mfcc=40)
for idx in range(5):
    audio, label = esc50_dataset[idx]
    # do something with mfcc feature, label
    print(audio.shape, label)
    # [feature_dim, length] , label_id