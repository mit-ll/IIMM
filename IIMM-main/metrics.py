import numpy as np


def cos_sim(images, text, image_labels):
    labels = text[image_labels.astype('int32').to_numpy(),:]
    return (images*labels).sum(axis=1).mean()

def gap(images, text):
    i = np.mean(images, axis = 0)
    t = np.mean(text, axis=0)
    return np.linalg.norm(i-t)

def uniformity_text(images, text, labels):
    a = np.matmul(text, images.T)
    for k, j in enumerate(labels):
        a[j,k] = np.nan
    return np.nanmean(a, axis=1).mean() 

def uniformity_image(images, text, labels):
    a = np.matmul(text, images.T)
    for k, j in enumerate(labels):
        a[j,k] = np.nan
    return np.nanmean(a, axis=0).mean()

def inter(e, batch=True, batchsize=10000):
    if batch:
        s = 0
        count = 0
        batches = np.ceil(len(e) / batchsize)
        for i in range(int(batches)):
            batch = e[batchsize*i:batchsize*(i+1), :]
            a = np.matmul(batch, batch.T)
            vals = a[np.tril_indices_from(a,k=-1)]
            s += np.sum(vals)
            count += len(vals)
            for j in range(i+1, int(batches)):
                _batch = e[batchsize*j:batchsize*(j+1), :]
                vals = np.matmul(batch, _batch.T)
                s += np.sum(vals)
                count  += vals.size
        out = s / count
    else:
        a = np.matmul(e, e.T)
        out = np.mean(a[np.tril_indices_from(a,k=-1)])
    return out

def IIMM(images, text, labels):
    intra_images = uniformity_image(images, text, labels)
    inter_modal = inter(images)
    return np.mean([intra_images, inter_modal])