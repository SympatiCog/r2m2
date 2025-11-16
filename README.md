# R2M2
Regional Registration Mismatch Metric<br>
---
Voxelwise assessment of the fit between an NxNxN neightborhood surrounding each voxel within an MRI image and its corresponding template space. <br>
``` python
    Arguments:
        tmplt:   template image
        reg_img: image registered to the template
        mask:    the template image's mask
        radius:  search radius to compute the r2m2 metrics
    Returns:
        image dictionary with r2m2 values.
        Each of:
            MattesMutualInformation
            MeanSquares
            Correlation
```
This code base serves as a proof of concept for the approach, and is not intended for production. Following PoC, the expensive operations currently implemented in native python will be rewritten in a faster approach - Julia, Numba, etc.
