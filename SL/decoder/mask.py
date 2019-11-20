def masked_softmax(arr, mask):
    '''
    Masks out invalid orders using the given mask and performs softmax over remaining

    Args:
    arr - array to mask and softmax
    mask - binary mask of the same size as arr.

    Returns:
    Softmax'ed result of applying given mask to h_dec
    '''
    tf.nn.softmax(tf.math.multiply(arr, mask))
