import torch


def add_i_and_u(i, u, i_total, u_total, batch_idx):

    # Sum i and u along the batch dimension (gives value per class)
    i = torch.sum(i, dim=0) / i.shape[0]
    u = torch.sum(u, dim=0) / u.shape[0]

    if batch_idx == 0:
        i_total = i
        u_total = u
    else:
        i_total += i
        u_total += u

    return i_total, u_total


def get_mean_iou_per_class(i_total, u_total):

    i_total = i_total.type(torch.FloatTensor)
    u_total = u_total.type(torch.FloatTensor)

    mean_iou_per_class = i_total / u_total

    return mean_iou_per_class





