import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from utils import load_data, scRNADataset, adj2saprse_tensor, Evaluation, embed2file
from scGNN import GraphMAE, LinkModel

def main():
    exp_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\Specific Dataset\mHSC-L\TFs+1000\BL--ExpressionData.csv"
    tf_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\Specific Dataset\mHSC-L\TFs+1000\TF.csv"
    target_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\Specific Dataset\mHSC-L\TFs+1000\Target.csv"
    
    train_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\1000训练\xunlianSpecific\mHSC-L 1000\Train_set.csv"
    val_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\1000训练\xunlianSpecific\mHSC-L 1000\Validation_set.csv"
    test_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\1000训练\xunlianSpecific\mHSC-L 1000\Test_set.csv"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
    target = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)
    
    feature = torch.from_numpy(feature).to(device)
    tf = torch.from_numpy(tf).to(device)
    data_feature = feature

    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values

    train_load = scRNADataset(train_data, feature.shape[0], flag=False)
    adj = train_load.Adj_Generate(tf.cpu().numpy(), loop=False)
    adj = adj2saprse_tensor(adj).to_dense().to(device)

    train_data_t = torch.from_numpy(train_data).to(device)
    val_data_t = torch.from_numpy(validation_data).to(device)
    test_data_t = torch.from_numpy(test_data).to(device)

    model = GraphMAE(
        input_dim=feature.size()[1],
        num_hidden=256,
        num_layers=2,
        output_dim=16,
        device=device,
    ).to(device)

    optimizer_pre = Adam(model.parameters(), lr=1e-3) 
    pre_epochs = 200  

    print("开始 GraphMAE 预训练...")
    for epoch in range(pre_epochs):
        model.train()               
        optimizer_pre.zero_grad()   
        loss, _ = model(data_feature, adj) 
        loss.backward()
        optimizer_pre.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Pre-training Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}")

    print("预训练完成，正在提取节点特征...")
    model.eval()                    
    with torch.no_grad():           
        a = model.encode(data_feature, adj)
    print("特征提取完毕！")

    linkmodel = LinkModel(input_dim=256, output_dim=32, hidden_dim=64).to(device)
    optimizer = Adam(linkmodel.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    epochs = 1000
    batch_size = 128
    pos_weight = torch.tensor([1.5]).to(device)  

    print("开始 LinkModel 训练...")
    for epoch in range(epochs):
        linkmodel.train()
        running_loss = 0.0
        
        for train_x, train_y in DataLoader(train_load, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            train_x = train_x.to(device)
            train_y = train_y.to(device).view(-1, 1)
       
            pred = linkmodel(a.data, train_x, adj)
            loss_BCE = F.binary_cross_entropy_with_logits(pred, train_y, pos_weight=pos_weight)
            
            loss_BCE.backward()
            torch.nn.utils.clip_grad_norm_(linkmodel.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss_BCE.item()

        scheduler.step()
        linkmodel.eval()
        with torch.no_grad():
            score = linkmodel(a.data, test_data_t, adj)
            score = torch.sigmoid(score)

        AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data_t[:, -1], flag=False)
        
        print('Epoch:{} train loss:{:.4f} AUC:{:.3f} AUPR:{:.3f}'.format(
            epoch + 1, running_loss, AUC, AUPR))

if __name__ == '__main__':
    main()