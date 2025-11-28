import torch
import pandas as pd
import numpy as np
from PytorchTools import scRNADataset, load_data
from scGNN import GraphMAE, LinkModel
from utils import Evaluation, adj2saprse_tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# 配置参数
exp_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\STRING Dataset\mESC\TFs+1000\BL--ExpressionData.csv"
tf_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\STRING Dataset\mESC\TFs+1000\TF.csv"
target_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\STRING Dataset\mESC\TFs+1000\Target.csv"
train_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\1000训练\xunlianSTRING\mESC 1000\Train_set.csv"
val_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\1000训练\xunlianSTRING\mESC 1000\Validation_set.csv"
test_file = r"D:\python\GENELink-main\Dataset\Benchmark Dataset\1000训练\xunlianSTRING\mESC 1000\Test_set.csv"

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据加载
    print("Loading data...")
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    
    tf_data = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
    
    train_data = pd.read_csv(train_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values

    train_load = scRNADataset(train_data, feature.shape[0], flag=False)
    adj = train_load.Adj_Generate(tf_data, loop=False)
    
    adj = adj2saprse_tensor(adj).to_dense().to(device)
    
    train_data_tensor = torch.from_numpy(train_data).to(device)
    test_data_tensor = torch.from_numpy(test_data).to(device)
    feature_tensor = torch.from_numpy(feature).to(device)

    # 模型初始化
    print("Initializing models...")
    model = GraphMAE(
        input_dim=feature_tensor.size()[1],
        num_hidden=256,
        num_layers=2,
        output_dim=16,
        device=device,
    ).to(device)
    
    # 修复：直接使用feature_tensor而不是调用encode方法
    # 因为GraphMAE需要先训练，这里我们直接使用原始特征或简单处理
    with torch.no_grad():
        encoded_features = model.encoder(feature_tensor, adj)
    
    linkmodel = LinkModel(
        input_dim=256,
        output_dim=32, 
        hidden_dim=64
    ).to(device)
    
    # 训练设置
    optimizer = Adam(linkmodel.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    pos_weight = torch.tensor([1.5]).to(device)
    
    # 训练循环
    epochs = 100
    batch_size = 128
    
    print("Starting training...")
    for epoch in range(epochs):
        linkmodel.train()
        running_loss = 0.0
        batch_count = 0
        
        for train_x, train_y in DataLoader(train_load, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            train_x = train_x.to(device)
            train_y = train_y.to(device).view(-1, 1)
            
            pred = linkmodel(encoded_features.data, train_x, adj)
            
            loss_BCE = F.binary_cross_entropy_with_logits(pred, train_y, pos_weight=pos_weight)
            
            loss_BCE.backward()
            
            torch.nn.utils.clip_grad_norm_(linkmodel.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss_BCE.item()
            batch_count += 1

        scheduler.step()
        
        # 评估
        linkmodel.eval()
        with torch.no_grad():
            score = linkmodel(encoded_features.data, test_data_tensor, adj)
            score = torch.sigmoid(score)

        AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data_tensor[:,-1], flag=False)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}, '
                  f'Train Loss: {running_loss/batch_count:.4f}, '
                  f'AUC: {AUC:.3f}, '
                  f'AUPR: {AUPR:.3f}')

if __name__ == "__main__":
    main()