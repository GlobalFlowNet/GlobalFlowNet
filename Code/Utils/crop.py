import numpy as np
import cv2



def crop_mask(mask):
    aspect = mask.shape[0] / mask.shape[1]
    if aspect > 1:
        size = mask.shape[0]
    else:
        size = mask.shape[1]
    mask_resize = cv2.resize(mask.astype(np.int8), (size, size), interpolation=cv2.INTER_NEAREST)
    S = (mask_resize > 0).astype(np.int64)
    nr, nc = S.shape
    for r in range((nr-2), -1, -1):
        for c in range((nc-2), -1, -1):
            if S[r, c]:
                a = S[r, c+1]
                b = S[r+1, c]
                d = S[r+1, c+1]
                S[r, c] = min([a, b, d]) + 1

    am = S.argmax()
    r = am // S.shape[1]
    c = am % S.shape[1]

    if aspect > 1:
        return r, c/aspect, S[r, c], S[r, c]/aspect
    else:
        return r*aspect, c, S[r, c]*aspect, S[r, c]



