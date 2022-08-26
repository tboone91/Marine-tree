## SUMMARY & USAGE LICENSE

This dataset contains more than 160k annotated images divided into 60 classes organised in a hierarchy-tree structure (up to five levels depth). The images correspond to  benthic organisms and substrate from different locations around the world of subtidal and intertidal reef communities. The images were meticulously collected by scuba divers using the RLS (Reef Life Survey) methodology and later annotated by experts in the field. After this, according to the annotation, we cropped the individual object around the annotation point generating 161,185 crops. Additionally, this dataset can be used as two subsets (subdatasets) according to its climate: Tropical and Temperate. Please, see the following table for further details.

<table align="center">
<thead>
  <tr>
    <th>Hierarchy level</th>
    <th>Temperate</th>
    <th>Tropical</th>
    <th>Combined</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2"></td>
    <td>118,637 Images</td>
    <td>42,548 Images</td>
    <td>161,185 Images</td>
  </tr>
  <tr>
    <td colspan="3",align="center">Classes</td>
  </tr>
  <tr>
    <td>$\ell_1$</td>
    <td>2</td>
    <td>2</td>
    <td>2</td>
  </tr>
  <tr>
    <td>$\ell_2$</td>
    <td>10</td>
    <td>9</td>
    <td>10</td>
  </tr>
  <tr>
    <td>$\ell_3$</td>
    <td>37</td>
    <td>34</td>
    <td>38</td>
  </tr>
  <tr>
    <td>$\ell_4$</td>
    <td>44</td>
    <td>38</td>
    <td>46</td>
  </tr>
  <tr>
    <td>$\ell_5$</td>
    <td>50</td>
    <td>52</td>
    <td>60</td>
  </tr>
</tbody>
</table>
<br></br>
<p align="center">
  <img src="/deakin/figures/Fig1_marine.jpg" width="550" title="hover text">
  <figcaption style="text-align: center" >Fig.1 Examples of images in Marine-tree (top) and a snapshot of one root-to-leaf branch (bottom) </figcaption>
</p>

<br></br>

<p align="center">
  <img src="/deakin/figures/world_map.png" width="550" title="hover text">
  <figcaption style="text-align: center" >Fig.2 Location of RLS campaigns: Red dots represent approximate location which belongs to the countries in  blue. </figcaption>
</p>

<br></br>

<p align="center">
  <img src="/deakin/figures/annotation_process.png" width="550" title="hover text">
  <figcaption style="text-align: center" >Fig.3 An example of one RLS diver photoquadrat used to build the dataset. The image is divided in a 5x5 grid in the cropping process and the cells with annotations are kept</figcaption>
</p>

<br></br>

Neither Deakin University nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user may not state or imply any endorsement from the Deakin University.

* The user must acknowledge the use of the data set in publications resulting from the use of the data set (see below for citation information).

* The user may not redistribute the data without separate permission.

The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from Asef Nazari at the Deakin University.

* If you have any further questions or comments, please contact Tanya Boone Sifuentes (tanyaboone20@gmail.com).

## CITATION

To acknowledge the use of the dataset in publications, please cite the following paper:

* Marine-tree: A large-scale hierarchically annotated dataset for marine organism classification, Sifuentes, Tanya Boone and Nazari, Asef and Razzak, Imran and Bouadjenek, Mohamed Reda and Robles-Kelly, Antonio and Ierodiaconou, Daniel and Oh, Elizabeth, (CIKM 2022).


## DETAILED DESCRIPTION OF THE DATA FILE

This dataset consists of three files which contains:

* marine_images.zip : all images
* 6 files with the format <train/test>\_labels\_<comp/temp/trop>.csv which contains:
  * fname (filename) for all images in marine_images.zip
  * scheme8 label (from Squiddle+) for each level of taxonomy
  * text label (e.g. "Biota") for each level of taxonomy
  * class_level for each level of taxonomy

## HOW TO LOAD
```python
from deakin.edu.au.data import get_Marine_dataset

dataset = get_Marine_dataset(output_level='all',image_size=(64, 64),subtype='Tropical',batch_size=128)

```

