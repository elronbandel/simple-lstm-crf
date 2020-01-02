from data_helper import TagData
from crf import LSTMCRF
from dataset import loader
from torch.optim import Adam, SGD
import torch
import datetime as dtm


def train_crf(model, epochs, optimizer, train_loader, eval_loader, device=None):
    device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt_str = str(optimizer).replace('\n ', ',')
    logging(f'Training - epochs:{epochs}, optimizer:{opt_str}, device:{device}')
    for epoch in range(epochs):
        avg_loss = None
        tg, tt = 0, 0
        for i, (data, target) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            loss = model(data, target)
            avg_loss = loss.item() if avg_loss is None else (0.99*avg_loss + 0.01*loss.item())
            loss.backward()
            optimizer.step()
            model.eval()
            pred = model(data, target)
            tg += sum(torch.eq(torch.LongTensor(p), t).sum() for p, t in zip(pred, target))
            tt += sum(map(len, data))


        # Eval
        model.eval()
        with torch.no_grad():
            eg, et = 0, 0
            for data, target in eval_loader:
                pred = model(data, target)
                eg += sum(torch.eq(torch.LongTensor(p), t).sum() for p, t in zip(pred, target)).int()
                et += sum(map(len, data))

            logging('Done epoch {}/{} ({} batches) train accuracy {:.2f}, eval accuracy {:.2f} avg loss {:.5f}'.format(
                epoch+1, epochs, (epoch+1)*train_loader.__len__(), float(tg) / tt, float(eg) / et, avg_loss))

def logging(message):
    print('{} {}'.format(dtm.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], message))


if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = TagData('pos')
    model = LSTMCRF(vocab_size=len(data.words), embed_dim=100, hidden_dim=100
                    , num_tags=len(data.tags), bidirectional=True, num_layers=2, device=device, dropout=0.5).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    train_crf(model, 10, optimizer, loader(data, 'train', 1000), loader(data, 'dev', 10), device=device)