# Worm-Kymograph
Code for worm tracking and kymograph creation. 
Coordinated neural patters are essential for life. While you read this, your breathing and blinking patterns are controlled by groups or neurons that generate coordinated output without coordinated input. (Unless you think about breathing every day!)
How exactly these neural circuits do that is not entirely known. The nematode _C. elegans_ is an excellent model organism in neuroscience and neurobiology. It has a suite of genetic and optogenetic tools with which we can probe these neural mechanics. In addition, it is the first organism to have its wiring diagram tracked throughout development, thanks to our lab in Toronto. (Witvliet et al., 2021)

We are interested in how the worm generates and modulates rhythmic neural behaviour. In particular, we look at the process of feeding in the worm. Feeding involves many coupled and rhythmic motions such as chewing and swallowing, both of which are demonstrated in the worm. 

![6v2cam](https://user-images.githubusercontent.com/30483987/192911114-0914aced-9a57-4814-ba23-c312167fbff5.gif)

A video of the worm eating. Notice how it "chews" food from outside to the middle, and "swallows" food to its stomach.

To study these neural mechanics in detail, one must quantify the feeding process, which is what this pipeline aims to do. It takes in an h5 video recording which has been formatted in a certain way by members of our lab. 

<img width="401" alt="Raw" src="https://user-images.githubusercontent.com/30483987/192909816-822a8e1b-ef41-45b6-95be-42620a8c1aef.png"> <img width="407" alt="image" src="https://user-images.githubusercontent.com/30483987/192910356-ea4d30c4-1548-4ad2-b3db-ffc96879e564.png"> 

The segmentation map is created with a CNN that we've trained on previous data. From the segmented map, we fit the centreline to a spline function so that we can integrate the pixel intensity along the centreline of the worm. 

<img width="425" alt="image" src="https://user-images.githubusercontent.com/30483987/192910031-055c2b35-1328-4c10-aabf-fd4bf8c64905.png">

The pixel intensity is integrated along the line normal to the centreline, to generate a food intensity waveform, which represents the food distribution along the worm. Sub pixel intensity values are interpolated with billinear interpolation and we use a block diagonal matrix to concurrently do the interpolation and integration.

<img width="362" alt="image" src="https://user-images.githubusercontent.com/30483987/192910441-5de38ad2-6d12-4ff5-9181-c498865f8d26.png">


The waveform is generated for each frame, and then aligned with subsequent frames to get a food transport kymograph. Y-axis is length along the body, X-axis is time. Downward sloping lines to the right indicate food is moving __down the body__ and __forward in time__. We can see the worm chewing and swallowing from just looking at the graph. What you are looking at is the first time that we've quantified food moving in an animal _in vivo_!

<img width="468" alt="image" src="https://user-images.githubusercontent.com/30483987/192910457-ca4b7e80-5f65-436f-833b-83df97f4c45d.png">


