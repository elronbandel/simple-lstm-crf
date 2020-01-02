import torch
from crf import LSTMCRF

if __name__ == '__main__':

    model = LSTMCRF(2, 10, 20, 3)
    s1w = torch.LongTensor([0, 1, 0, 1])
    s1t = torch.LongTensor([1, 0, 1, 2])
    s2w = torch.LongTensor([0, 1, 0, 1, 0])
    s2t = torch.LongTensor([1, 0, 1, 0, 2])
    seqs, tags = [s1w, s2w], [s1t, s2t]
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in range(10000):
        model.train()
        optimizer.zero_grad()
        loss = model(seqs, tags)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(loss.item())
            model.eval()
            print(seqs)
            print(model(seqs, tags))