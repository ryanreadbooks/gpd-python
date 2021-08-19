import torch
import torch.utils.data as torch_data
import torch.nn.functional as F

from gpd.core import *

def evaluate():
    threshold = 0.5
    in_channel = 12
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = GraspImageDataset(root='test/images', train=False, channel=in_channel)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)
    print(f'test_loader len = {len(test_loader)}')
    classifier = GPDClassifier(input_chann=in_channel, dropout=False)
    classifier.load_state_dict(torch.load('model/model-chann12.pth'))
    classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        eval_total_loss = torch.tensor(0., device=device)
        for batchidx, test_data in enumerate(test_loader):
            repr_test, label_test = test_data
            repr_test, label_test = repr_test.to(device), label_test.to(device)
            size = repr_test.shape[0]
            classifier.eval()
            pred = classifier(repr_test)
            out = pred.sigmoid()
            eval_total_loss += F.binary_cross_entropy_with_logits(input=pred, target=label_test.reshape(-1, 1))

            prediction = out.detach().cpu().reshape(-1,)
            prediction = torch.where(prediction >= threshold, 
                                    torch.tensor(1.), torch.tensor(0.))

            # print(out, label_test)
            # compute accuracy
            res = torch.logical_not(torch.logical_xor(prediction, label_test.cpu()))
            accuracy= res.sum() / torch.tensor(len(res), dtype=torch.float32)
            print(f'mean test loss at batchidx {batchidx} with size {size} is {eval_total_loss.item() / size}. Accuracy = {accuracy}')


if __name__ == '__main__':
    evaluate()