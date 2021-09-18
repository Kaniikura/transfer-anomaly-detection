# Transfer Anomaly Detection using ImageNet Pre-trained CNN
## Usage
### Dataset Setup
* `ShanghaiTech AD`  
You need to download [ShanghaiTech](https://onedrive.live.com/?authkey=%21AMqh2fTSemfrokE&cid=3705E349C336415F&id=3705E349C336415F%2172436&parId=3705E349C336415F%215109&o=OneUp) manually.
* `MVTec AD`  
Data preparation is done automatically when the code is executed.
## Environment Setup
### Prerequisites
- cuda >= 11.0
### Anaconda
```
conda env create --file=environment.yaml
```
### Docker
```
docker build ./docker -t transfer-ad
``` 

## Example
### Train & Eval on MVTec AD
```bash
python main.py --class_name='bottle' --img_size=456 \
--fe_name='efficientnet-b5' --extract_blocks='0,1,2,3,4,5,6' \
--reducer_name='pca' --percentile_components=0.5 --negated_pca='True' \
--estimator_loss='none' --estimator_name='knn' --n_neighbors=1 \
--epochs=10 --gpus=1
```

## Integration with Weights & Biases
Create an account on [Weights & Biases](https://wandb.ai/). Then, make sure you install the latest version.
```
pip install -U wandb
```
Locate API_KEY in the user settings and activate it:
```
wandb login <API_KEY>
```
