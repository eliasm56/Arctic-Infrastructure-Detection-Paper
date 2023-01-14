# Arctic-Infrastructure-Detection-Paper
Code for paper:

```
Manos, E.; Witharana, C.; Udawalpola, M.R.; Hasan, A.; Liljedahl, A.K. 
Convolutional Neural Networks for Automated Built Infrastructure Detection 
in the Arctic Using Sub-Meter Spatial Resolution Satellite Imagery. 
Remote Sens. 2022, 14, 2719. https://doi.org/10.3390/rs14112719
```

Link: https://www.mdpi.com/2072-4292/14/11/2719

Citation
```
@Article{rs14112719,
AUTHOR = {Manos, Elias and Witharana, Chandi and Udawalpola, Mahendra Rajitha and Hasan, Amit and Liljedahl, Anna K.},
TITLE = {Convolutional Neural Networks for Automated Built Infrastructure Detection in the Arctic Using Sub-Meter Spatial Resolution Satellite Imagery},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {11},
ARTICLE-NUMBER = {2719},
URL = {https://www.mdpi.com/2072-4292/14/11/2719},
ISSN = {2072-4292},
ABSTRACT = {Rapid global warming is catalyzing widespread permafrost degradation in the Arctic, leading to destructive land-surface subsidence that destabilizes and deforms the ground. Consequently, human-built infrastructure constructed upon permafrost is currently at major risk of structural failure. Risk assessment frameworks that attempt to study this issue assume that precise information on the location and extent of infrastructure is known. However, complete, high-quality, uniform geospatial datasets of built infrastructure that are readily available for such scientific studies are lacking. While imagery-enabled mapping can fill this knowledge gap, the small size of individual structures and vast geographical extent of the Arctic necessitate large volumes of very high spatial resolution remote sensing imagery. Transforming this &lsquo;big&rsquo; imagery data into &lsquo;science-ready&rsquo; information demands highly automated image analysis pipelines driven by advanced computer vision algorithms. Despite this, previous fine resolution studies have been limited to manual digitization of features on locally confined scales. Therefore, this exploratory study serves as the first investigation into fully automated analysis of sub-meter spatial resolution satellite imagery for automated detection of Arctic built infrastructure. We tasked the U-Net, a deep learning-based semantic segmentation model, with classifying different infrastructure types (residential, commercial, public, and industrial buildings, as well as roads) from commercial satellite imagery of Utqiagvik and Prudhoe Bay, Alaska. We also conducted a systematic experiment to understand how image augmentation can impact model performance when labeled training data is limited. When optimal augmentation methods were applied, the U-Net achieved an average F1 score of 0.83. Overall, our experimental findings show that the U-Net-based workflow is a promising method for automated Arctic built infrastructure detection that, combined with existing optimized workflows, such as MAPLE, could be expanded to map a multitude of infrastructure types spanning the pan-Arctic.},
DOI = {10.3390/rs14112719}
}
```


