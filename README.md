# information-retrieval-systems
Implementation of tf-idf model and LSI-Model 

`why do heart doctors favor surgery and drugs over diet ?`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.2   | 0.1   | 0.05  | 0.04  |
| recall    | 0.026 | 0.026 | 0.026 | 0.051 |
| f1        | 0.045 | 0.041 | 0.034 | 0.045 |

R-precision: 0.05<br>
AP 0.052

-------------

`berries to prevent muscle soreness`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.8   | 0.4   | 0.25  | 0.12  |
| recall    | 0.057 | 0.057 | 0.071 | 0.086 |
| f1        | 0.107 | 0.1   | 0.11  | 0.1   |

R-precision: 0.1<br>
AP 0.286

-----------

`fukushima radiation and seafood`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.8   | 0.4   | 0.25  | 0.2   |
| recall    | 0.114 | 0.114 | 0.143 | 0.286 |
| f1        | 0.2   | 0.178 | 0.182 | 0.235 |

R-precision: 0.2<br>
AP: 0.308

-----------

`diabetes`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.8   | 0.9   | 0.9   | 0.82  |
| recall    | 0.007 | 0.015 | 0.035 | 0.069 |
| f1        | 0.013 | 0.03  | 0.06  | 0.128 |

R-precision: 0.302 <br>
AP: 0.685

-----------

`heart rate variability`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 0.2   | 0.1   | 0.05  | 0.02  |
| recall    | 0.125 | 0.125 | 0.125 | 0.125 |
| f1        | 0.154 | 0.111 | 0.071 | 0.034 |

R-precision 0.125 <br>
AP: 0.023

MAP 0.138 <br>
gemittelte 11-point AP 0.271

-----------
# LSI
`why do heart doctors favor surgery and drugs over diet ?`

| metric    | @5 | @10 | @20   | @50   |
|-----------|----|-----|-------|-------|
| precision | 0  | 0   | 0.05  | 0.04  |
| recall    | 0  | 0   | 0.026 | 0.051 |
| f1        | 0  | 0   | 0.034 | 0.045 |

R-precision: 0.0256  <br>
AP: 0.0359

-----------
`berries to prevent muscle soreness`

| metric    | @5 | @10 | @20   | @50   |
|-----------|----|-----|-------|-------|
| precision | 0  | 0   | 0.05  | 0.02  |
| recall    | 0  | 0   | 0.014 | 0.014 |
| f1        | 0  | 0   | 0.022 | 0.017 |

R-precision: 0.0286 <br>
AP: 0.023

-------------
`fukushima radiation and seafood`

| metric    | @5 | @10 | @20 | @50 |
|-----------|----|-----|-----|-----|
| precision | 0  | 0   | 0   | 0   |
| recall    | 0  | 0   | 0   | 0   |
| f1        | 0  | 0   | 0   | 0   |

R-precision: 0.0 <br>
AP: 0.014

----------------------

`diabetes`

| metric    | @5    | @10   | @20   | @50   |
|-----------|-------|-------|-------|-------|
| precision | 1     | 1     | 0.95  | 0.78  |
| recall    | 0.008 | 0.017 | 0.032 | 0.067 |
| f1        | 0.017 | 0.033 | 0.062 | 0.122 |

R-precision: 0.353 <br>
AP 0.679

-----------------
`heart rate variability`

| metric    | @5 | @10 | @20 | @50 |
|-----------|----|-----|-----|-----|
| precision | 0  | 0   | 0   | 0   |
| recall    | 0  | 0   | 0   | 0   |
| f1        | 0  | 0   | 0   | 0   |

R-precision: 0.0
AP: 0.0015

MAP 0.0032
gemittelte 11-point AP 0.1507

<img src="/images/11AP.png">
<img src="/images/precision.png">
<img src="/images/recall.png">
<img src="/images/f1.png">

## Results Vektorraum Model

### k = 5

Enter query: ***berries to prevent muscle soreness***
Enter top k: 5

--------------------------------------------------

Found 5 documents in 0.003 seconds:
[(3462, 0.07658248065407205), (2271, 0.06594936320608871), (3461, 0.06586469335399849), (3171, 0.0629004812916967), (1569, 0.05132740136021903)]

--------------------------------------------------

Enter query: ***diabetes***
Enter top k: 5

--------------------------------------------------

Found 5 documents in 0.003 seconds:
[(1706, 0.021278137451919008), (2147, 0.019054574282802723), (2763, 0.018605861244109255), (2144, 0.018126255048801176), (4100, 0.01683532918599409)]

--------------------------------------------------

Enter query: ***why do heart doctors favor surgery and drugs over diet***
Enter top k: 5

--------------------------------------------------

Found 5 documents in 0.007 seconds:
[(3755, 0.23010858574744653), (2322, 0.11196931772971298), (1248, 0.07467971023088608), (3180, 0.07467971023088608), (2284, 0.06259474372937598)]

--------------------------------------------------

Enter query: ***fukushima radiation and seafood***
Enter top k: 5

--------------------------------------------------

Found 5 documents in 0.003 seconds:
[(4359, 0.07105224028987228), (5101, 0.045362611303617795), (3386, 0.0394431668043326), (3630, 0.03939223406240791), (3608, 0.03543091645553737)]

--------------------------------------------------

Enter query: ***heart rate variability***
Enter top k: 5

--------------------------------------------------

Found 5 documents in 0.004 seconds:
[(4315, 0.04359670261996087), (2724, 0.039813104535551175), (5026, 0.03911318934898814), (1285, 0.03646200331306186), (2111, 0.034698664708823)]

--------------------------------------------------

### k = 10

Enter query: ***berries to prevent muscle soreness***
Enter top k: 10

--------------------------------------------------

Found 10 documents in 0.003 seconds:
[(3462, 0.07658248065407205), (2271, 0.06594936320608871), (3461, 0.06586469335399849), (3171, 0.0629004812916967), (1569, 0.05132740136021903), (1414, 0.04941876776127938), (1466, 0.046419209798040637), (3728, 0.042460327020492517), (4100, 0.0379746649751207), (3231, 0.03585422158247033)]

--------------------------------------------------

Enter query: ***diabetes***
Enter top k: 10

--------------------------------------------------

Found 10 documents in 0.004 seconds:
[(1706, 0.021278137451919008), (2147, 0.019054574282802723), (2763, 0.018605861244109255), (2144, 0.018126255048801176), (4100, 0.01683532918599409), (4987, 0.016661454389336004), (5237, 0.015937016403686804), (2846, 0.015734320073408584), (1541, 0.015205322449207507), (5281, 0.014562482509253986)]

--------------------------------------------------

Enter query: ***why do heart doctors favor surgery and drugs over diet***
Enter top k: 10

--------------------------------------------------

Found 10 documents in 0.007 seconds:
[(3755, 0.23010858574744653), (2322, 0.11196931772971298), (1248, 0.07467971023088608), (3180, 0.07467971023088608), (2284, 0.06259474372937598), (3757, 0.04887773169080731), (5311, 0.047698081233740336), (3708, 0.045851735532402495), (5008, 0.04373876983257952), (4365, 0.04371663270012957)]
--------------------------------------------------
Enter query: ***fukushima radiation and seafood***
Enter top k: 10

--------------------------------------------------

Found 10 documents in 0.002 seconds:
[(4359, 0.07105224028987228), (5101, 0.045362611303617795), (3386, 0.0394431668043326), (3630, 0.03939223406240791), (3608, 0.03543091645553737), (3632, 0.03354546657068116), (3635, 0.03308799174062538), (3604, 0.03282552096812939), (4968, 0.03135765020015419), (3619, 0.029705396692459335)]

--------------------------------------------------

Enter query: ***heart rate variability***
Enter top k: 10

--------------------------------------------------

Found 10 documents in 0.004 seconds:
[(4315, 0.04359670261996087), (2724, 0.039813104535551175), (5026, 0.03911318934898814), (1285, 0.03646200331306186), (2111, 0.034698664708823), (4791, 0.03379134470703545), (1257, 0.033353784666539), (2763, 0.03292259191902688), (713, 0.03225045795963902), (5248, 0.030680140512600967)]

--------------------------------------------------
