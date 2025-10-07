# NailongIdentification
A deep learning model for identifying Nailong with high accuracy.

This model achieves over **98% accuracy** and can reliably distinguish **Nailong** from **Konote Fujita** (and any other items!).

## Demo
<table>
  <tr>
    <td align="center">
      <img src="pictures/1.jpg" alt="Nailong" width="300"><br>
      <p><b>Nailong</b></p>
    </td>
    <td align="center">
      <img src="pictures/2.jpg" alt="Konote Fujita" width="300"><br>
      <p><b>Konote Fujita</b></p>
    </td>
  </tr>
</table>

### Just For Fun!
Watch the demo video on Bilibili: [Demo Video](https://www.bilibili.com/video/BV18gDdY3EV6/) (CN)

## Model
- **Network:** VGG11
- **Training Visualization:** Includes loss and accuracy plots [Loss and Accuracy](model/results/VGG11-Nailong_plot.png)

## Dataset
Download the dataset from: [HuggingFace Nailong Dataset](https://huggingface.co/datasets/XiC1/nailong-dataset)

## Music & Inspiration
Inspired by and music sourced from: [Douyin Video](https://www.douyin.com/video/7433341884916862260)

## Environment Requirements
- **GPU** is required for training and inference.
- Python packages: `Python3.x`, `os`, `torch`, `torchvision`, `PIL`, `matplotlib`, `tkinter`, `pygame`

## How to Use
1. Download the entire project repository.
2. Ensure all environment dependencies are installed.
3. Run `api.py` to start the application.
