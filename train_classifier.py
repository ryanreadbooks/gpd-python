import torch
import torch.utils.data as torch_data
import torch.optim as optim
import torch.nn.functional as F

from gpd.core import *


def train_classifier():
    in_channel = 12
    n_epoch = 5000
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = GraspImageDataset(root='test/images', train=True, channel=in_channel)
    dataloader = torch_data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
    
    classifier = GPDClassifier(input_chann=in_channel, dropout=False)
    classifier.to(device)
    classifier.train(True)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.8)
    print('starts training...')
    for epoch in range(n_epoch):
        batch_loss = torch.tensor(0., device=device)
        for batchidx, data in enumerate(dataloader):
            repr, label = data
            repr, label = repr.to(device), label.to(device)
            pred_out = classifier(repr)
            loss = F.binary_cross_entropy_with_logits(input=pred_out, target=label.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            batch_loss += loss
        if epoch % 50 == 0:
            print(f'[After epoch {epoch}/{n_epoch}], loss = {batch_loss.item()}')
        scheduler.step()
    print('saving the model...')
    torch.save(classifier.state_dict(), 'model/model-normal-10000.pth')


if __name__ == '__main__':
    train_classifier()