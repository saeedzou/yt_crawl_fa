import time
import csv
import argparse
import sys
import re
import random
import os
import yt_dlp
import torch
import subprocess
import string
import re
import librosa
import numpy as np
from pathlib import Path
from jiwer import wer, cer
from parsnorm import ParsNorm
from util import make_video_url
from nemo.collections.asr.models import ASRModel
from tqdm import tqdm

EXCLUDED_CHANNEL_IDS = ['UCJEJkoZVLfBCbxWSC8XmIMA',
 'UCxWhvRqBPnYAFe9ULG4xWOQ',
 'UCgweSBy42APwNp7BOo15vbw',
 'UCbLBmmyhWSJZO7-7CwnxVZw',
 'UC8nRV2doa0Xmjpso2uDLrpA',
 'UCIawj8WCPViLQENKM59kyWg',
 'UCRRNHRlQxF0rb_MFQlJCSBA',
 'UCd3u9d6XjfzyV7m2_9W6BiA',
 'UCLA5a-zLn7Iqsn_MRF7xlPQ',
 'UC_PlW1XLNETrWO7xn2DYqSg',
 'UCTU-dlTMVcfgKOoQRSkFugA',
 'UC5afHpaMSpeLmUFo93AP5iQ',
 'UC7UlGp2RH0D0Tpt4FAKsYzg',
 'UCblWqgXbjPAfMtvX1CmZ-og',
 'UC501xv-cmrFQ22I6_hs-KEQ',
 'UCYQWo0RTu42_7h3gfrCh3ow',
 'UCOIutjo5AsF-OM6_mT0gw-w',
 'UCbTKuebM_JA66IsdZR5Opcg',
 'UCD2KvhGhZbRqX_NaDH44KLA',
 'UCcWXrqJRMTRMGeTkX-d1Beg',
 'UCjQCk4jivOjoWMYWUAR07FQ',
 'UCmKpLU6-_US4mXCavOCDPCA',
 'UCqRL9AMzLRxAYGiQHcVY0-A',
 'UCWvQhzYgHHws1egPPWnQfdw',
 'UCorbBAIXzUCp1WMBDzSEAUg',
 'UCMXtnIVdCjQ6-j_2pFTDWrQ',
 'UCi-RTT-gBFej3qBVXCQXeGg',
 'UCTKVcIZi1ZkWfuLg-LKNCtw',
 'UCyTBi8M075V4EF0NvJ1Bpng',
 'UCaD97sRPcB2_Wdcwu0_w3lQ',
 'UCc_8l0cplOR7J6QzIS2aI7Q',
 'UCMyVETyiHXQmGorL-Vht9JQ',
 'UC139xYlPA99c1090SnWkoSg',
 'UCFrVef_NY6Tt7YDviddN89A',
 'UCLxUUku0o91T9nurLCuNw8g',
 'UCjb1suYZDVqovT_dDQbpWYQ',
 'UCRWRR1X8ibxDAei-GAQLyvg',
 'UCCmpeONqHfElttj89gfNlBw',
 'UCH6RNVbWkiUkl15yB24C9Qw',
 'UCC0crkECeSn2fS2AcenO4cA',
 'UConNkzFgmEygaNfq33OBd0Q',
 'UCK3yc4rh0D7Dvu7TYXWNuSA',
 'UC_uoI5e67rO00oxktjvL55g',
 'UCqiPfSQWNJenrTBAVCZVNow',
 'UCN78pAZy6iglAb3Xv0isQyQ',
 'UC8xebZ-xgI-vf4M9D2q3k7w',
 'UCGfGfOjAdd0ahSiG4Z14qig',
 'UCthlxe_gHPosix6pLbQr_YQ',
 'UCDvxToLVx545jFXCgThOtTw',
 'UCqRJZfQjEoVi7s4tyKIGWrA',
 'UCWyn9Imgo_W-2qOGmDAewNA',
 'UCgr2hHWQ3cKtLJPec2eWUrg',
 'UCI0QvWyqH2wh-2HsDUl6OjQ',
 'UCg5cqCyCaSQjoJEF5G4OwdQ',
 'UCtCC4f-MXOZ4PR26FsgNqKA',
 'UCWHEjv-OmN2J6mkAJ051OJg',
 'UCe5liv0ZvcelpvuVQa6YALw',
 'UCmt9zcKUwM-3UEWZLvYC9nQ',
 'UCLE4h9ts2zLVkMbtwrg8Eyw',
 'UCsf0zH_9BgZlNmv2OGkZMIw',
 'UCpM5D9omGPgncDAVPQOb_0w',
 'UCVNXB6iChrAjlILvtWXvIMQ',
 'UCGaJ1f8NSK_Y8MnGydHlZ0A',
 'UC1moSlhc3HTLH1foAE3E_BA',
 'UC6-dtAqzSUa41JN-oNwZCLA',
 'UCEwxAWaXt-6VVsgq3HZrtSw',
 'UCK8d8vqrgPcrdjFOv7deavg',
 'UCCMRXGelE4Q3lsah7ZqdsLA',
 'UCz2_XrLntRUqS-172XTN44w',
 'UCiGG7wOuJn_e_x3D2fE6v4w',
 'UCgLeMwwPHasYb4--zrcAQYw',
 'UCBu3S8z_gum6wEiK4oWVNlA',
 'UCKq_4KPTbD3Zl-SeF1dfAEA',
 'UCVz5-O0rsZv8pTlZeHtnWLQ',
 'UCKCL_cZRS5-bnydcPSZSWRA',
 'UCcq-32SO0JNaDskt1Dw0Pjw',
 'UC1bOcOug9BNDs17AyBzj-tg',
 'UCHZk9MrT3DGWmVqdsj5y0EA',
 'UC6hW6hwuhdULLzE3zpGynHQ',
 'UCgeWLAuIFyfpKy0eAYDVjHQ',
 'UCmRNXk6hpY7T8ct9gNIyMhQ',
 'UCbOdYVEeH2oPnYtVA4r2Kyg',
 'UCyrEnWQzKvWV2XjYtyx6RXA',
 'UC0bMxe-AeNzTL_nIFf3BebQ',
 'UCvNwbIfk87xewXzDHefdEPw',
 'UCQVGiDLBAyNB2Mvgsqh1WZA',
 'UCv5UZen5xrh_q7PdqwrsD9Q',
 'UCrCJBpFiWCWP-aT8n1B1L2g',
 'UCRePYYG1kR3lyxacTlDt5Eg',
 'UCofEFDso-NmL4axVWAFAANA',
 'UChWB95_-n9rUc3H9srsn9bQ',
 'UCvuQ9ttcMS-szPQ1XUFsOZg',
 'UCaAVQYHKMbY_eybBLbqgOYA',
 'UCsESFnPCZ6G3iBBciLG4yZg',
 'UC3LawKxD0_ZWAKcrwIikNpw',
 'UC4Ub2ZqIHcFC0X9q-QZln9A',
 'UC6bxWJulB-NMWvje7Y3k7BA',
 'UCu31Ij7Mf8dgy_sPOgUeV_g',
 'UCwhskmdWYz7lpCp_uRka7sQ',
 'UCdsBwylFszwKnhL7URueB2Q',
 'UCHy8QSSJ1gXu-zbDlwwsE-w',
 'UCJ9d0dg3Yx7xEWlcfKCM3QA',
 'UCcLXqA4CHUgxxt5KWTiO0Ow',
 'UCIueilRpFpg3dLtW-e4BgMA',
 'UCIsS-8wQCApt3E_UJIhKssg',
 'UCSlgmchrzAPFADzG3vjN7Cw',
 'UCpZAuWTnrQ7xC7kGyQE4Wog',
 'UC8zvy-GRqvlxHGIJ64iDejA',
 'UC3GkUOPm6ssYW6UQk1P0Cvg',
 'UCLc4b29_L7eYYvhoxoUW0jg',
 'UCLPc4eGvzkvMO3FuaF2_W4g',
 'UC0_R-e0XTs307kFUrmxKF8g',
 'UC5C4SG6EPMARNg9ueF9p-YA',
 'UChc2VZucm1G4qEMLyAdd_Uw',
 'UCv-o3pxm3y0M6KRnZGGYbAA',
 'UChtqBGfrGaX4NrraRaXsGaA',
 'UCkkedgv0XAqZoDos3AMMABg',
 'UC2fRLzOCkisSfNQWnVnGxcQ',
 'UCdqxSCV0JpfA-jMtBLdtK6A',
 'UC3_3SKISt4r668w5XKeUcCg',
 'UC0EzA7xABPAs4zqFZcRDvMg',
 'UC__3xFxVVPqLihNZwdH4nCw',
 'UCFv3Kwp9M5hTlfOglb-fHsQ',
 'UCmgsjjFDPLKDq1FaGYT7zAQ',
 'UCqJHR37XbfxfcbhRXWfo_pQ',
 'UCHvUw_0R3NQ9c__siFLFC1Q',
 'UCXHdNtE3dr2WFgdvFF66gig',
 'UCFHoXoyvKKpVvZNpxydy19Q',
 'UCEXvgES4tZwyzn2cf-6eFnw',
 'UClvrmK9fdbJbbKz9-BZ3c3A',
 'UC5By6SJv2VbCuq54aCZKmgg',
 'UCCVDDvROLQkeHNGxJ5FHCVA',
 'UChO_FjlcFGw6LOmQ5AXivAA',
 'UC5IsU7vjFN5U-ma5l0SfgiA',
 'UC-oz-WwAfLWbuOuvHIluG8w',
 'UCjeDYAGCpn9-Kyj2T7rQ2DQ',
 'UCYtIj9wOGX9O8fe-q_WyfIA',
 'UCTxD9VUwXgysFrpKkulct6Q',
 'UCxmr92bafw_h7FhLBUpo8gg',
 'UC29BJYNsAUkGKaN3IN7MnKA',
 'UC1dToGrt7bWrJ-Nqqy05ceg',
 'UCszzm33jU3ncxm5gdrUUIjQ',
 'UCZgQTPO27tddTU_MvMOO4gg',
 'UCPFskIzFclgXhmLWbwSL5uA',
 'UCBlUMV5CVTeeB8oJkw9eDBw',
 'UC48O1qqj9F1gYGvsjGvE2DA',
 'UC7QU3gdA6wl7GXbG5WMGv2A',
 'UCIabVeJ4Cb1WofcUwwU4GHw',
 'UCv0yRKS3gnaBZFfzdBhivFQ',
 'UClQjdb1wYITTDMNBH2k_Wcw',
 'UC-dk_S-YA4oNGiRyevoi0fA',
 'UCd56VD5DSUsVCY-AdxdMBwg',
 'UCyQ7PpUAy-FoUA0CDusXYig',
 'UCGLVhMF9kLx0nRQ9wquUsXw',
 'UC6iN8omCNhk9RVzIziyfepw',
 'UCQEqPNpIgTw3yzN8r9u0MsA',
 'UCFiYjI6yi9xqoF1u5ZS0SqA',
 'UCTOUUfBH96TRqAHT13vM3Eg',
 'UCXL4oq2DZ-mkQE1by_ZiTbA',
 'UC2y81UEI3gvpnVH_avkvDWQ',
 'UC1t1VahaNpUwI-xy58bPbHw',
 'UCNhstD5nB8oWgrnR8OKcdOA',
 'UCctjlScohDIOCKKALxLIyhA',
 'UCPKqXz2TNvl6Vzco2bBv-oQ',
 'UCrEOEHrtXaZv_34xty_mwcQ',
 'UCLqLTbZvsPsdzX8Uf-ZV52A',
 'UCly_Nmr4z2j3Wugi9Ov4Zmg',
 'UCflljYZDSP_FQrnFvXh8MkQ',
 'UC8Vey_xjcIJqRgel--YwyQw',
 'UCsNd7V7rQT2Qdzl1nP7y0Dg',
 'UCE-BF2-I5Oc2srI-H2gcAXA',
 'UCUKEFTT2GjlNbHjqsn4cexA',
 'UCH_Bp7VV_o3Ur21vww3ktcQ',
 'UCrhMfpGSNX0yNxTt3t8aZMQ',
 'UCwdRCJx8w9TxHmkbh7Pa9KA',
 'UCZ7IN4m4Z5bRm2cWxBVOxZg',
 'UCeG5opSfsaKI59suiaCtFlw',
 'UCPGn18BoNgjH6vG_IlzOkvQ',
 'UC2cA7ueXE8d_5GzdzVTj0aA',
 'UCu3BULxmErRNNrEg2paR4gA',
 'UCuSS7Q8O6Wv1SGQVAs1Uvgg',
 'UCdGuwiX9vsD_cLZZ9n4avPQ',
 'UCCrXxTy4nDvA514_vexMz3w',
 'UC5ULPe6zBuV7tN0GkKz9I0g',
 'UCkZ5ctGBOe7bXm3CuPQ48OQ',
 'UCyyRWb-YPfWUGl_GD5vTJqA',
 'UC-9hF4gu5i4jBdusI0SmzlQ',
 'UCRSjjoOw9DeljDCMoIFdyvw',
 'UC-7VTj3nEz2RPBa2qa9Cf5A',
 'UCMYy1LJqXYo5JcE8v7HKdZA',
 'UCZrqHYP33fcDNOilYX1ollQ',
 'UCxbiY_k6lofbgHGRGvvwSCQ',
 'UCJVTec-58WNxjYHF-hhEy9A',
 'UCvPH1e4hjvc9LSCkkr-FZFA',
 'UC9IzZm4enrFdABOMI1LBTYA',
 'UC7gQR0nEmO_mcnGr4i0XqiA',
 'UCaHuW_EW_7hzUIcJtBrWjdQ',
 'UCbILSnU1x2Nh1LCt7N7FIZQ',
 'UC1tg2CjBFW6Oh2uwfYyBM2w',
 'UCzZv4A_zJ5r5Ci9INm_QD9A',
 'UCxDS3mVo3HST02TQquSJvVQ',
 'UCts_dlAHuIdP81DBkqfW_OA',
 'UCLFoY1liRRYx-VY2xXrykIw',
 'UCD4SNlqK36o3DdUCeSltQ6w',
 'UCzcsMxLnDE3OpY3n3av8iHA',
 'UCjTKBp44yeTvq48TY5CTxUA',
 'UC1URV0USV0QvlbAT9PImqrA',
 'UCPEfAfhqbsLrf108Yn37Eog',
 'UCClAWsF4UVO-lb0KA8aqHHQ',
 'UCn2idumr8SyO2-wErv_75lg',
 'UCIDZomoRLOCbRcApi-8YgFQ',
 'UC6uu0lMr8PMYiuGm_dwe0rA',
 'UCpmQZZ0i89c8AirfrIk_t8A',
 'UCat6bC0Wrqq9Bcq7EkH_yQw',
 'UCIN32v8cGU4HTmJvUNEacdA',
 'UCoyt2LLBfcc_NeZfxm1Kx_w',
 'UCU1bB2uRgDxiR9vXTEYnbTQ',
 'UCbENF-6whN-uSsrHL3Di9vw',
 'UCQvn1frpIuWWWoWfpJR1asg',
 'UC37NO-y7SQmmLAFFdqYiYow',
 'UCxJUULmM3tWXMuq838dc_1A',
 'UCIrU-TPq3WfGeVNZnn8sW5g',
 'UC2wUcT51vbHsGYdQif5EUTw',
 'UCN-8mIGprQrxDV-wdtxVj-w',
 'UC_tOsJgQZ-dEFw5DzhQ6xpw',
 'UCMOaKNarH5TlZitTz6PTbYA',
 'UCR75e74FoTh9rt3BNH9sAOw',
 'UC2KUuwZlvVzkx7nmZTGdxgA',
 'UCYtmv6jsW5mgJ71oLmLwM4A',
 'UCQlriihf9taSwo2ke-KGNBA',
 'UCtFh-kUGFCDLBI1FoSbd5kw',
 'UCkzVrG8Pyqeog2qiT8goMag',
 'UCEqm7ZP24EHX7nFlerSDQFg',
 'UCOsVwFsP2X-k0xXKCBwX4Ug',
 'UCeE6q1h-Tid5rso3pTsXkPg',
 'UCJwC86Ui5YC93Q0wSYdrcOA',
 'UCsMl-W1Ve2MwRhIWcSMjHiA',
 'UCCsSgwmvanyiVpsYHeyKNTA',
 'UC6Ab-DVf4lyk-zktnMpDVaA',
 'UCOLrhONl267Sr-vMRecNXRw',
 'UCYNJXTMsYtp5sxQ8iQoAkAw',
 'UCLPDT5Jpa_FTugdh1gI48Rg',
 'UCGICgH_kwrD6eMuQKEPJdNg',
 'UCUrzRPVSTRYsovjNovfcc_A',
 'UCa-ZgzdNQYDOet7t_yrcPHw',
 'UC9hKyDQywmhDqKYuuHbUCRA',
 'UCf67DKdLhpFW-7c7FZre2Ww',
 'UCAZPEcTJZFw73jgrp1_XqNw',
 'UCCLCieDdRxiw-B5wnRSLdgQ',
 'UCxA29iWdyCo6VvozB_OBj0Q',
 'UCsdIwOQT1B8fHogCTHiDByQ',
 'UCdz7qUouvVMUuY6_MS7vVHQ',
 'UCFw3nQAR2DiPPDcAlfkXsDw',
 'UCo0nUsn8TS-UQNi21iBc1VA',
 'UC2MNGImXpckrbfOQdHGmimA',
 'UCGvHfyQfBlRxmYCKAQr7d_A',
 'UCEEOHeU_xm4wwGDClGQiUVA',
 'UC8s2re8LFyku1hw4ja6hjQQ',
 'UCsN2Md5UDvJFJrRBn2YOdYQ',
 'UCilx5uLc4Es6GTEW1CZ2jaQ',
 'UCvkZh97nemeYk0SVzJ9kcXA',
 'UCg5cqCyCaSQjoJEF5G4OwdQ',
 'UCfyJkwgLNnKMryjLjhLlFeA',
 'UCcewDOFwcMB3hkIS_gC72fQ',
 'UCnUdm0u-2FRffBnxQYHuTHA',
 'UC-9Wnqc5XSTX7mr5sAbD63w',
 'UCYMNGh4s97QIuzDN40QRtPw',
 'UCRrE8Zqaq9dsfuwqqPZZ38A',
 'UCpqZfhGwp9nplhx7msClc3w',
 'UCjiaUn5KSMDFPcM_Mbey9ew',
 'UCNsY22Odo-vnjE_XlJpsK-g',
 'UCWeDqvErMYRN5rDxQqhJMsw',
 'UCzK5ODik5zcorEB-6WfN0Aw',
 'UCzH_7hfL6Jd1H0WpNO_eryQ',
 'UC50RmHjcDRXRBFqi3HmWnsw',
 'UCUh4trNr5-u6UCqXBv-uQ0w',
 'UCEFRS2UMQwcCZpfHhcj6c0w',
 'UC4c6mVy1FL5d4hCwW5mMgfQ',
 'UCeQ6lVmCzUEyGrQeJ3KqYqQ',
 'UCOxBzhdB_seyRIuwRJmGBhg',
 'UClxSLe8-ZF1K1alqiDvni5Q',
 'UClxSLe8-ZF1K1alqiDvni5Q',
 'UC-mKi1k--PTmqagL-8aWZdQ',
 'UC8TMa4T3mz47P3t2gcHg1Bg',
 'UCq2XBsnbOzGhcni6mJ4DorA',
 'UCj_cdl3vFs9OqwSBGDYbzeA',
 'UCMYTerZpsU2gTEbPWM8PllA',
 'UCGiHtKmiFJSWKTfXe38Z88g',
 'UCAKLIpB4c9FRlk50U9rpbyw',
 'UC5L5Py1GbMHNs11f8CCdCSA',
 'UCpnMrI2_oYPiztn8H0eA8Iw',
 'UCtOenTMbr-_4xCcTCCLaZiA',
 'UCLgFV67YSYCILKBeHZY2Tuw',
 'UCFk1aSJx4GYCtmfNskr2x_Q',
 'UCy6WM5_SkYXNgrsavllmf_Q',
 'UCqqivLON7RXXclXBNo5RbKA',
 'UC4gOpC1FexzUhljL75ZOF4Q',
 'UCf8AsrNrb2hYJ5zsxYAU3dg',
 'UC_cVjxXhxGQnkzpj0YNL5lQ',
 'UC2lXiX0OaMlIWn5AotYjpOw',
 'UCDFQwxd5HtjIS9IS7sLw3hw',
 'UCWgvLJIHNiSVT1YJOeVh4pQ',
 'UC45z3byO7n7_XbrguWGRaJw',
 'UCq9eBr7OortDJz4fGARKTlQ',
 'UC2yGPxeZcn9f3rifBBtWBcw',
 'UCTaOK6-1gwcLhS_xDGIefsA',
 'UCwnGlZmlHaLbaAnKd24PO9w',
 'UC36Lu8eWcnhmhfYrbMdQHNQ',
 'UCcBUIM59SWoIl01gnWMy_mw',
 'UC_MyHvz65A-xzdi9PVmGIzw',
 'UCSudXpJE9KMxesNA_jPgkPw',
 'UCtS6I8HYq0kCczkMxrSmYXA',
 'UCrgwpzal3QrmmmhpYegzWbg',
 'UC-Z_J-XnLjW5G2KF5S-31Xg',
 'UCz5I1VybGOubYxin_TGcKww',
 'UCDPi-RJtm6Vob60LXKlU8iw',
 'UC4f8Me7v-5nUyj_j01S5asA',
 'UCnE88-vstou5gmF4lWI2Z-g',
 'UCbuanM0pkNF70whYnfo7duA',
 'UCYw-zz-mh5T1dFHf_6hzHxw',
 'UC24MSMZj4ljXbhTnBCMtosg',
 'UCOSSx27FK3UN7RMkeZpVgLw',
 'UCNpGrPGFK2HNq_TJn-8NDhQ',
 'UCqqd050wW1f_kDUECp1nU7A',
 'UC7BfUYMUwddfnwCNvq29QpQ',
 'UCEdjL1kbZ0K8nFu519vE5LQ',
 'UCAlUDRPJAayxmmvsyFgdlhw',
 'UC0nqfrQfoUVwxKYEH9lXUQA',
 'UCAgdG_Y0CyqbcbxQ4u4u3lA',
 'UCiSffMi8g7_wXQXRCQa7XTA',
 'UCgIn1v6GCd15E8niXYmaEhA',
 'UCSKPhxfwgW-FJgLJiM7MoDw',
 'UCM-jROc6CjyxXxvwY1B3-eQ',
 'UCzg4ErA1_-lWBC59SVike2g',
 'UCQiwqOaVlmKxTJKgR2K9DOQ',
 'UCSUSuGu3vwJWiruNJ12-C8g',
 'UChxIQP4D05EBdRHX57_mSFQ',
 'UC-jtPLc1cu6H78pK-HnBo-w',
 'UChiyq4qjnAWMNhwPu2KL4yg',
 'UCSxCDdGuvR_3eoLYk2F5w9Q',
 'UCZGvidZW7-YTVG7PY3-xw1A',
 'UCcbDvE-J9_eUtAdOlF_St6g',
 'UCgjmFb37-miTopvG9zc0aaA',
 'UCwXf3hQOrG71aiwguykt44Q',
 'UCovhHa2CvNYKg9UWvudKMJQ',
 'UC-CiY7kSHTttCrLBaPelwgQ',
 'UCRyX8eZrbIFVQ-davXDmjKw',
 'UCbflNrzfMdoHRsIRJlC-2iQ',
 'UC40DnK4_Up_YwQ4HV5WUB_w',
 'UCdnu04hIoKpfvweOakGvK_A',
 'UC4SuZNKgciL6xPdJZK1lGlw',
 'UCxKxwN9VqFU6yIntPxi0F6g',
 'UCw67dxqjR5euO7dmW2MMV1A',
 'UC02VpPhZkp8ZIJwVkebM8tg',
 'UCVZuFvBxN0XE8dK6kpSxL1Q',
 'UCKfRga0bG3l8L8-JzXj7fgA',
 'UCkPaaJrmdeygudLbtTaPUhg',
 'UCQedtHzD6d2KZgQu67ZZAAA',
 'UCqhFKsZsSX9-Fu07lVUY5GA',
 'UCyaGHRMxONmzEOzTEIdjb0Q',
 'UCQofX0gggpRsBAqpDEk0q5w',
 'UCqYCssczpdf9f9oNJPQKiIQ',
 'UCVjt1H4u5z3Ya7_hxHGus6w',
 'UCbeu4GSzDkJ7R5Ngch2SV8w',
 'UCznRng1thHdiJnaTE9gZ6aA',
 'UCI_uzpH6AdZ5EnjrQ231rnA',
 'UCOnCHSl69gwwBNnLd8A_T6g',
 'UClCTljqZHyHYTp2j3i8VNJw',
 'UCFMX9SlZz_hvJh5wMSqTZUA',
 'UCFfiMeJTUeUmn-_r_YyTBOQ',
 'UCrsK_-sKEbBU1y-XZc2sD3A',
 'UCYLtf0GG-SSLo2v1cb51Mkg',
 'UCGGGOdL3NzM-HAq-53cr_fA',
 'UCC3IRtuuoZBSUe_BfwlhEyg',
 'UCs9nBfW_nyDiRjqpQoajWNA',
 'UCeb1M62YnyUwA-nN02NriGA',
 'UCUQ2HMpBAgDy2cH1ikXDgUQ',
 'UCbOzB9d4RCxiK2XW3CFfAYw',
 'UCmESRi_kDkZNljd-hiY9umQ',
 'UCPURVs56y2GkbAgicRCE3KQ',
 'UC7xic815sE6NosnfZ8wt42A',
 'UCQRLcm6aCHlY7k8equRjZbA',
 'UCFVsjJtaN1W7fqBvxDaK1_Q',
 'UCIDN5oNxPtLuPuk9hqpvgUw',
 'UCuCIM_dfiVholrxyqDA1oxw',
 'UCgteArGT_05NR3oOuQb7VFg',
 'UCwiUhqsrJGnU089XWj92jeg',
 'UCO-YPv4Yd4d_6QCyU1tp-_w',
 'UCV462-3nPby72c5WczXj_Sg',
 'UCsrkHushFpSeTMsUVzu0ixg',
 'UC941Wo1FbrD1MDSGP2-ANKA',
 'UCVTB6PSsgZMVparYnPvu93A',
 'UC2F9R4bSsTaU0e7Oz2zkL1Q',
 'UCxVENdN3Vr-2MpUGcOYntLw',
 'UCGJYRj8ambMYl1_t5imzS9g',
 'UCtIZ87u4Bq0kBJv130GUcOg',
 'UCVU5-v9U7l6_6i0Ar04zWDA',
 'UC5NigWeRhscj0CWsTSfnkpA',
 'UCBEE9AnjqsKYAPsrk4HMwSQ',
 'UCM4cnv_tGwAAMXtcQDr7s1g',
 'UCpLvEuHiB1-HlfIJcjiL2lQ',
 'UCWDstwzkyyW5foDZT6pSKig',
 'UCVRB3RZeHda1FHpK4rz8y5w',
 'UCTTokcxhXbve2FGlRUPOTnw',
 'UCHfKCw-OTDXJ6-bgnyybQPw',
 'UCCHkdN_D92IZ7WDCrWvI3pg',
 'UC2ErIIyvDA6jEwAsqrjPyig',
 'UCmgQY8yg73ftqFkOEnUmeIQ',
 'UC_Ps08YoFn8bX7mC7e0w8Vw',
 'UC6XGQp_CbgAtXe-i42kjz8A',
 'UCzx5i7cPwt-Bgson-SRK0Ow',
 'UCrDDOcZ5UwBRa5kJxATzBeQ',
 'UC2QkBREXupaiLskEH1WVFNg',
 'UCjlZNUDWYSAsRpy1DnkkqcA',
 'UCpLyN2sOKXsOdzXyHDEUd1w',
 'UCC8wIK3115nedNU3qZaXrww',
 'UCQVLyGWJIM3SHFDB0Am-Rdg',
 'UCLjGwsNeTUr5ZiwlpaMM0bw',
 'UCE2VhHmwp3Y_wZvgz_LJvfg',
 'UCJB6_QkPGejY_PFx1RWq0ZA',
 'UCJ3hAYNGZPVElGL0tapYtyw',
 'UCuKBL5yb6pUcFEn59-sptKA',
 'UCMWOoUbSTkqTfAWY41aTaPg',
 'UCwEh-V1cVRyq42o0jNehSnQ',
 'UCounWv9wcAkmn1US7WACWAw',
 'UC_muvaiTMTLrIesOYIMIXRQ',
 'UCOoxb9u59nNMxH2t2RDwohw',
 'UCsT0YIqwnpJCM-mx7-gSA4Q',
 'UCRKmcyXeDqptP9zL8ZOiGXw',
 'UCLSRJD8sryHZWsrn6sKsTvg',
 'UCJhtAsVZILr8o3iEM-6Ldig',
 'UCZcHwkoZK_2nd8mAD15Is2Q',
 'UC6AaHE6Aij58-36_sRx7pQQ',
 'UCL1zER_t4b_fUphxbOTfmjw',
 'UCu_aKTTdG3oZjPA4RJN9XRg',
 'UCvYWBMdqoXDIrLaL8EldbQg',
 'UCCF6ACdVMPtkrkKU0L7FHpg',
 'UCaj8a3FWpJ64hyZd5M4bG6g',
 'UCNe_VAP5oEszkg-QCxowatg',
 'UC4L_9u8U_RDYGz2SW7Q4D8g',
 'UCg--Uz7iJB3yGvEakaL2zmw',
 'UCxelJIiHJoGR270g8FqtfJA',
 'UCCn_0vWsa35jSTRNlX15hzQ',
 'UCBVUT0Q8cCXkm5_3vQ-Uhlg',
 'UCJuAKBlUyGxWZG9aQHQQEwg',
 'UC-r3VY8LVyNK2dCew-L67cQ',
 'UCMNk8c1KI4FAa9zDt1q1lsA',
 'UCBqLoTi2FeITULTyalLIEIQ',
 'UC9uDm-WfL0xPqRxmyx-Bb4Q',
 'UCX0PqnbNlQL3MZ9ueBp7bLQ',
 'UCJE75Eeu8GM_wdVnMD9syYA',
 'UC4SQb28v1pS4FR5PJ2CQ78Q',
 'UCbscehIo0evtXIlpUjJH8rw',
 'UCttfDeGMwUxPjnlsKagcwKw',
 'UCkGRCFk4NPQMXcQj9N5q-YQ',
 'UCF8hjTCujXfi1oA_NnMIOyA',
 'UC64rwLSEudq71Qa9IAexy0Q',
 'UC_JqqM-LjYk7ew4tWc-MlMA',
 'UCFOGvucMi6eg7wXrUJ9IRbw',
 'UCNv6I5rA345ZW6qSHUFYwog',
 'UCjURxbTOxy663W8u7tUQHCw',
 'UCtyrlhup5rzGi0VsSjgWKLg',
 'UC5gx0Pyl3UZ9YztuoZ-jFAQ',
 'UCDF7x2EYvBwoQA6kQG0C67A',
 'UCMM0Sy4oLJtw-ywqTpkGBRg',
 'UC9oBEuVBMV3EbOvCCt1naWA',
 'UC4IPXv7mHV08eSmqg8RmwVA',
 'UClrf-ogHB5ewrWtyxyFJR0A',
 'UCdrcyLjUN7_ruQnwZ5fVCrQ',
 'UCpQbrDu0bd0ejElOaTPnwrQ',
 'UCBJh-s0jj7f96NejSMu2HIg',
 'UCjwwXrLucLjB3zWt4QiOA9Q',
 'UCNKie6lYVCKfjRcTbzy72fg',
 'UCYIIz-xX6fMjq32M3SfwKtA',
 'UCgBFo2szwBlHxgleVFSm24w',
 'UChJDoitU8BCdhxeAeFOQIHg',
 'UCa_U3xRzNeqKAREIZCTCfSQ']

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve video metadata and subtitle availability status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("lang", type=str, help="language code (ja, en, ...)")
    parser.add_argument("videoidlist", type=str, help="filename of video ID list")
    parser.add_argument("--outdir", type=str, default="sub", help="dirname to save results")
    parser.add_argument("--checkpoint", type=str, default=None, help="filename of list checkpoint (for restart retrieving)")
    return parser.parse_args(sys.argv[1:])

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    # convert to mono
    waveform = librosa.to_mono(waveform)
    # Normalize and convert to float32
    if waveform.dtype == 'int16':
        waveform = waveform.astype('float32') / 32768.0
    elif waveform.dtype == 'int32':
        waveform = waveform.astype('float32') / 2147483648.0
    elif waveform.dtype == 'uint8':
        waveform = (waveform.astype('float32') - 128) / 128.0
    else:
        # If already float32, ensure no further normalization is done
        waveform = waveform.astype('float32')

    return waveform, sample_rate

def load_model(model_path:str="/content/drive/MyDrive/stt_fa_fastconformer_hybrid_large_dataset_v30.nemo"):
    model = ASRModel.restore_from(restore_path=model_path)
    return model

normalizer = ParsNorm()
model = load_model()

def transcribe_chunk(audio_chunk, model):
    transcription = model.transcribe([audio_chunk], batch_size=1, verbose=False)
    return transcription[0].text

def transcribe_audio(file_path, model, chunk_size=30*16000):
    waveform, _ = load_audio(file_path)
    transcriptions = []
    for start in range(0, len(waveform), chunk_size):
        end = min(len(waveform), start + chunk_size)
        transcription = transcribe_chunk(waveform[start:end], model)
        transcriptions.append(transcription)

    # Combine all transcriptions and normalize the final result
    final_transcription = ' '.join(transcriptions)
    final_transcription = re.sub(' +', ' ', final_transcription)
    final_transcription = normalizer.normalize(final_transcription)
    
    return final_transcription

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    # convert to mono
    waveform = librosa.to_mono(waveform)
    # Normalize and convert to float32
    if waveform.dtype == 'int16':
        waveform = waveform.astype('float32') / 32768.0
    elif waveform.dtype == 'int32':
        waveform = waveform.astype('float32') / 2147483648.0
    elif waveform.dtype == 'uint8':
        waveform = (waveform.astype('float32') - 128) / 128.0
    else:
        # If already float32, ensure no further normalization is done
        waveform = waveform.astype('float32')

    return waveform, sample_rate


def is_english(text):
    """Returns True if the text contains more than 50% English alphabet characters."""
    # Count English alphabet characters
    english_chars = sum(1 for char in text if char in string.ascii_letters)
    # Total characters in the text
    total_chars = len(text)
    # Avoid division by zero
    if total_chars == 0:
        return False
    # Calculate percentage of English characters
    return (english_chars / total_chars) > 0.5

def count_common_punctuations(text):
    """Count common punctuation marks in text."""
    common_punctuation_marks = r'[؟،]'
    matches = re.findall(common_punctuation_marks, text)
    return len(matches)

def count_other_punctuations(text):
    other_punctuation_marks = r'[!؛:]'
    matches = re.findall(other_punctuation_marks, text)
    return len(matches)

def parse_timestamp(timestamp):
    """Convert WebVTT timestamp to seconds."""
    # Format: HH:MM:SS.mmm
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split('.')
    total_seconds = (int(hours) * 3600 + 
                    int(minutes) * 60 + 
                    int(seconds) + 
                    int(milliseconds) / 1000)
    return total_seconds

def calculate_subtitle_duration(subtitle_file):
    """Calculate total duration covered by subtitles."""
    total_duration = 0
    try:
        with open(subtitle_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[3:]
            for line in lines:
                if '-->' in line:
                    # Extract start and end times
                    start, end = line.strip().split(' --> ')
                    start_time = parse_timestamp(start)
                    end_time = parse_timestamp(end)
                    duration = end_time - start_time
                    total_duration += duration
    except Exception as e:
        print(f"❌Error calculating subtitle duration: {e}")
        return 0
    return total_duration

def extract_text_from_subtitle(subtitle_file):
    """Extract plain text from subtitle file, removing timings."""
    text = ""
    try:
        with open(subtitle_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[3:]
            for line in lines:
                # Skip timeline patterns and empty lines
                if not line.strip():
                    continue
                if re.match(r'^\d+$', line.strip()):
                    continue
                if '-->' in line:
                    continue
                # Add non-empty, non-timeline lines to text
                if line.strip():
                    text += line.strip() + " "
    except Exception as e:
        print(f"❌Error reading subtitle file: {e}")
        return ""
    return text.strip()

def extract_subtitle_text(subtitle_file: str) -> str:
    if not subtitle_file or not os.path.exists(subtitle_file):
        return None

    with open(subtitle_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    text = " ".join(
        line.strip() for line in lines[3:]
        if not line.startswith('WEBVTT') and not line.strip().startswith('0') and line.strip()
    )

    # Remove text between parentheses
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove text between square brackets
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove text between asterisks
    text = re.sub(r'\*[^*]*\*', '', text)

    # Remove emails
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '', text)

    # Remove URLs
    text = re.sub(r'\b(?:http[s]?://|www\.)\S+\b', '', text)

    # Normalize text (assuming normalizer is defined elsewhere)
    text = normalizer.normalize(text)

    return text

def download_video(video_id: str):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs('videos', exist_ok=True)
    output_template = f"videos/{video_id}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',         # Download best audio quality
        'outtmpl': output_template,
        'skip_download': False,             # Download the audio
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_file = ydl.prepare_filename(info).replace('.%(ext)s', info['ext'])

        return audio_file

def download_captions(video_id, lang):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs('subtitles', exist_ok=True)
    output_template = f"subtitles/{video_id}.%(ext)s"
    if lang == 'fa':
        lang = ['fa', 'fa-IR']
    else:
        lang = [lang]

    ydl_opts = {
        'outtmpl': output_template,
        'writesubtitles': True,             # Write manual subtitles
        'writeautomaticsubs': False,        # Explicitly disable auto-generated subtitles
        'subtitleslangs': lang,    # Only download Persian subtitles
        'skip_download': True,             # Download the audio
        'cookies': 'cookies.txt',
        'quiet': True,
        'list_subs': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

        # Look for Persian subtitle file
        subtitle_file = None
        for i in lang:
            potential_file = f"subtitles/{video_id}.{i}.vtt"
            if os.path.exists(potential_file):
                subtitle_file = potential_file
                break

        return subtitle_file, info

BOT_ERRORS = 0

def process_video(videoid, query_phrase, lang, excluded_channel_ids=EXCLUDED_CHANNEL_IDS):
    """Process a single video to get metadata, download Persian subtitles, and analyze punctuation."""
    url = make_video_url(videoid)
    entry = {
        "videoid": videoid,
        "videourl": url,
        "good_sub": "False",
        "sub": "False",
        "title": "",
        "query_phrase": query_phrase,
        "channel": "",
        "channel_id": "",
        "channel_url": "",
        "channel_follower_count": "",
        "upload_date": "",
        "duration": "",
        "view_count": "",
        "categories": [],
        "like_count": "",
        "punctuation_count": 0,
        "subtitle_duration": 0,  
        "cer": "",
        "wer": "",
    }


    try:
        # First request: Get subtitle info
        subtitle_filename, metadata = download_captions(videoid, lang)
        manu_lang = list(metadata['subtitles'].keys())
        has_subtitle = lang in manu_lang and len(manu_lang) < 5
        entry["sub"] = str(has_subtitle)
        try:
            entry.update({
                'title': metadata.get('title', ''),
                'channel': metadata.get('channel', ''),
                'channel_id': metadata.get('channel_id', ''),
                'channel_url': metadata.get('channel_url', ''),
                'channel_follower_count': metadata.get('channel_follower_count', ''),
                'upload_date': metadata.get('upload_date', ''),
                'uploader_id': metadata.get('uploader_id', ''),
                'uploader_url': metadata.get('uploader_url', ''),
                'duration': metadata.get('duration', ''),
                'view_count': metadata.get('view_count', ''),
                'categories': metadata.get('categories', []),
                'like_count': metadata.get('like_count', '')
            })
            if metadata.get('channel_id') in excluded_channel_ids:
                print(f"❕ Video {videoid} belongs to channel {metadata.get('channel', '')} - {metadata.get('channel_url', '')}")
                return entry
        except Exception as e:
            print(f"❌ Error updating metadata: {e}") 

        if has_subtitle:
            print(f"❕ Downloaded subtitle for video {videoid} to {subtitle_filename}")

            # Extract text and count punctuations
            if Path(subtitle_filename).exists():
                subtitle_text = extract_text_from_subtitle(subtitle_filename)
                common_punct = count_common_punctuations(subtitle_text)
                other_punct = count_other_punctuations(subtitle_text)
                punct_count = common_punct + other_punct
                entry["punctuation_count"] = punct_count
                
                # Calculate total subtitle duration
                subtitle_duration = calculate_subtitle_duration(subtitle_filename)
                entry["subtitle_duration"] = round(subtitle_duration, 2)  # Round to 2 decimal places
                if (entry["subtitle_duration"] > 10) and (not is_english(subtitle_text)) and (common_punct > 5 or other_punct > 1):
                    print(f"❕ Downloading and processing audio for video {videoid}")
                    print(url)
                    audio_file = download_video(videoid)
                    auto_transcription = transcribe_audio(audio_file, model)
                    manual_transcription = extract_subtitle_text(subtitle_filename)
                    word_error_rate = wer(manual_transcription, auto_transcription)
                    character_error_rate = cer(manual_transcription, auto_transcription)
                    entry["wer"] = word_error_rate
                    entry["cer"] = character_error_rate
                    if word_error_rate < 0.8 and character_error_rate < 0.2:
                        entry["good_sub"] = str(True)


    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing video {videoid}. stdout: {e.stdout}, stderr: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error processing video {videoid}: {str(e)}")
        if "Sign in to confirm you’re not a bot" in str(e):
            if BOT_ERRORS > 3:
                print("❌ Too many bot errors, exiting.")
                exit(1)
            BOT_ERRORS +=1

    return entry

def retrieve_subtitle_exists(lang, fn_videoid, outdir="sub", wait_sec=0.2, fn_checkpoint=None):
    fn_sub = Path(outdir) / f"{Path(fn_videoid).stem}.csv"
    fn_sub.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if provided
    subtitle_exists = []
    processed_videoids = set()
    if fn_checkpoint and Path(fn_checkpoint).exists():
        with open(fn_checkpoint, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subtitle_exists.append(row)
                processed_videoids.add(row["videoid"])

    # Load video ID list
    video_ids = []
    with open(fn_videoid, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.append((row["video_id"], row['word']))
    random.shuffle(video_ids)

    # Define fieldnames for CSV
    fieldnames = ["videoid", 
                  "videourl", 
                  "title", 
                  "good_sub", 
                  "sub",
                  "wer", 
                  "cer",
                  "channel", 
                  "channel_id", 
                  "channel_url",
                  "channel_follower_count", 
                  "view_count", 
                  "like_count", 
                  "uploader_id",
                  "uploader_url",
                  "upload_date", 
                  "duration", 
                  "punctuation_count", 
                  "subtitle_duration",
                  "query_phrase",
                  "categories", 
                  ]


    # Process videos
    for videoid, query_phrase in tqdm(video_ids):
        if videoid in processed_videoids:
            continue

        entry = process_video(videoid, query_phrase, lang)
        subtitle_exists.append(entry)

        if wait_sec > 0.01:
            time.sleep(wait_sec)

        # Write current result every 2 videos
        if len(subtitle_exists) % 2 == 0:
            with open(fn_sub, "w", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(subtitle_exists)

    # Final write
    with open(fn_sub, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subtitle_exists)

    return fn_sub

if __name__ == "__main__":
    args = parse_args()
    filename = retrieve_subtitle_exists(
        args.lang, 
        args.videoidlist, 
        args.outdir, 
        fn_checkpoint=args.checkpoint
    )
    print(f"Saved {args.lang.upper()} subtitle info, metadata, and punctuation counts to {filename}.")