### 总结 

- Parameters:
	`Parameters`是 torch.Tensor 的subclass, 特别之处在于:当和 Module 连用时(被assign Module 属性,会自动添加到 parameter list 中,封装成 Module.parameters 迭代器.
		属性:
		
      	data(Tensor)
        requires_grad(bool): Default: True
eg:
```python

	class Vgg16(nn.Module):
		 vgg_pretrained_features = models.vgg16(pretrained=True).features
		self.slice1 = nn.Sequential()	
		 for x in range(4):
		 	#此时添加模块后,会自动添加到parameter属性中
            self.slice1.add_module(name=str(x),module=vgg_pretrained_features[x])
         for param in self.parameters():
         	param.requires_grad = False
         	#param.data
```
- namedtuple

`from collections import namedtuple`
	
namedtuple是tuple的一种,不可修改,但类似字典,可用名字访问.eg:

```python
vgg_outputs = namedtuple("VggOutputs",['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
out = vgg_outputs(h_relu1_2,h_relu2_2,h_relu3_3,h_relu4_3)

print(out.relu4_3)
```
- IN and BN

假设形状为(B,C,H,W)
	
	IN对H,W计算,共B*C个
	BN对B,H,W计算,共C个