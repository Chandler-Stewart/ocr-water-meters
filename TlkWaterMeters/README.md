### Toloka Water Meters
The dataset consists of 1244 images of hot or cold water meters. Each image contains exactly one water meter.  The readings and the coordinates of digits are provided for each water meter image as well.
![](https://hsto.org/webt/pv/83/un/pv83unxagbpbdifzxexbfpuefwo.png)
The dataset is assembled by Kutsev Roman from the [TrainingData.ru](http://trainingdata.ru/)  company for an article on Habr.

### How the data was collected
The detailed process of collecting the dataset is described in the article  [“Create your own dataset for recognition of water meters on Yandex.Toloka”](https://habr.com/ru/company/ods/blog/469633/) (in Russian).

### Archive contents

## data.tsv
- photo\_name -- photo name
- value -- reading
- location -- display coordinates

## images
Directory with images of water meters

## masks
Directory with masks of water meters

## collage
Directory with collages of water meters