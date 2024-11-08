# NailongIdentification
A simple model for Identification of nailong based on deep learning  
Model reaches the accuracy of over 98%, and it can accurately tell Nailong from Konote Fujita！  
<style>
  figure {
    display: inline-block;
    text-align: center;
    margin: 0;
  }
  figcaption {
    margin-top: 5px;
    font-size: 14px; /* 可以调整字体大小 */
    color: #555; /* 调整颜色 */
  }
</style>

<table>
  <tr>
    <td>
      <figure>
        <img src="pictures/1.jpg" alt="Nailong" width="300">
        <figcaption>Nailong</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="pictures/2.jpg" alt="Konote Fujita" width="300">
        <figcaption>Konote Fujita</figcaption>
      </figure>
    </td>
  </tr>
</table>


### Just For Fun!!!  
#### Trained by network VGG11   
Process are visualized as figure including loss and accuracy  
## Dataset  
Dataset are download from https://huggingface.co/datasets/XiC1/nailong-dataset  
## Music  
Music and inspiration come from https://www.douyin.com/video/7433341884916862260
## Environment  
You need `Python3.x`, `os`, `torch`, `torchvision`, `PIL`, `matplotlib`, `tkinter`, `pygame`  
## How to use  
Download the whole project, make sure your environment is useful. Run `api.py`.  
