### 总结 

- NLLLoss(M,N):

	最大似然函数,接受的M是(batch,dim), N是(batch,)
	M * log (N) 
	会把N对应的真位置为1,其他位置为0 , 扩展后还是(batch,dim)
	
- CrossEntropyLoss()=log_softmax() + NLLLoss()
log_softmax = log( softmax )

- 学会用logging输出打印日志,以便查看
	- 导入logging
	- 配置logging, logging.basicConfig
	- 使用logging.info
```python
	import logging
	
	LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
	logging.basicConfig(filename='my.log', level=logging.DEBUG, 		format=LOG_FORMAT)
	
	logging.info('Train Epoch: {} [{} / {} ({:.0f}%)]\t Loss:{:.6f}'.format(epoch,idx * len(data),len(train_loader.dataset),100*idx / len(train_loader),loss.item()))
```

- 使用argmax 和 eq 
output为numpy, (batch,dim), labels为numpy, (batch,)
```python
pred = output.argmax(dim=1,keepdim=True)
#pred: (batch,1)
correct += pred.eq(labels.view_as(pred)).sum().item()
```
