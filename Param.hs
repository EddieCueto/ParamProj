-- file: Param.hs
-- try to calculate the number of parameters
-- of GooLeNet


-- Types for the fully conected layer and the

type Maps = Int
type KernSize = (Int,Int)
type InConv = Int
type SizeFC = Int

data Conv = Conv Maps KernSize (Maybe InConv)
            deriving (Show)

data FullyCon = FullyCon (Maybe Maps) SizeFC
                deriving (Show)

justExt (Just val) = val

convLayer (Conv maps kernsize inconv) = (maps * (fst kernsize) * (snd kernsize) * (justExt inconv)) + maps

convOut (Conv maps _ _) = maps

fcLayer (FullyCon maps size) = if maps /= Nothing
                               then ((justExt maps) * size) + (justExt maps)
                               else (size * size) + size


calcInMod (Conv maps1 kernsize1 inconv1) (Conv maps2 kernsize2 inconv2) (Conv maps3 kernsize3 inconv3) (Conv maps4 kernsize4 inconv4) (Conv maps5 kernsize5 inconv5) (Conv maps6 kernsize6 inconv6) =
  ((maps1 * (fst kernsize1) * (snd kernsize1) * (justExt inconv1)) + maps1) + ((maps2 * (fst kernsize2) * (snd kernsize2) * (justExt inconv2)) + maps2) + ((maps3 * (fst kernsize3) * (snd kernsize3) * (justExt inconv3)) + maps3) + ((maps4 * (fst kernsize4) * (snd kernsize4) * (justExt inconv4)) + maps4) + ((maps5 * (fst kernsize5) * (snd kernsize5) * (justExt inconv5)) + maps5) + ((maps6 * (fst kernsize6) * (snd kernsize6) * (justExt inconv6)) + maps6)

concMod (Conv maps1 _ _) (Conv maps2 _ _) (Conv maps3 _ _) (Conv maps4 _ _) = maps1 + maps2 + maps3 + maps4

-- The only 3 convolution layers

con1 = Conv 3 (7,7) (Just 64)
con2 = Conv 64 (1,1) (Just 64)
con3 = Conv 192 (3,3) (Just 64)

-- Constructing the nine inception modules

inCon11 = Conv 192 (1,1) (Just 64)
inCon12 = Conv 96 (3,3) (Just 128)
inCon13 = Conv 16 (5,5) (Just 32)
inCon14 = Conv 192 (1,1) (Just 32)
inCon15 = Conv 192 (1,1) (Just 96)
inCon16 = Conv 192 (1,1) (Just 16)
fstCon = concMod inCon11 inCon12 inCon13 inCon14

inCon21 = Conv fstCon (1,1) (Just 128)
inCon22 = Conv 128 (3,3) (Just 192)
inCon23 = Conv 32 (5,5) (Just 96)
inCon24 = Conv fstCon (1,1) (Just 64)
inCon25 = Conv fstCon (1,1) (Just 128)
inCon26 = Conv fstCon (1,1) (Just 32)
sndCon = concMod inCon21 inCon22 inCon23 inCon24

inCon31 = Conv sndCon (1,1) (Just 192)
inCon32 = Conv 96 (3,3) (Just 208)
inCon33 = Conv 16 (5,5) (Just 48)
inCon34 = Conv sndCon (1,1) (Just 64)
inCon35 = Conv sndCon (1,1) (Just 96)
inCon36 = Conv sndCon (1,1) (Just 16)
trdCon = concMod inCon31 inCon32 inCon33 inCon34

inCon41 = Conv trdCon (1,1) (Just 160)
inCon42 = Conv 112 (3,3) (Just 224)
inCon43 = Conv 24 (5,5) (Just 64)
inCon44 = Conv trdCon (1,1) (Just 64)
inCon45 = Conv trdCon (1,1) (Just 112)
inCon46 = Conv trdCon (1,1) (Just 24)
frtCon = concMod inCon41 inCon42 inCon43 inCon44

inCon51 = Conv frtCon (1,1) (Just 128)
inCon52 = Conv 128 (3,3) (Just 256)
inCon53 = Conv 24 (5,5) (Just 64)
inCon54 = Conv frtCon (1,1) (Just 64)
inCon55 = Conv frtCon (1,1) (Just 128)
inCon56 = Conv frtCon (1,1) (Just 24)
fthCon = concMod inCon51 inCon52 inCon53 inCon54

inCon61 = Conv fthCon (1,1) (Just 112)
inCon62 = Conv 144 (3,3) (Just 228)
inCon63 = Conv 32 (5,5) (Just 64)
inCon64 = Conv fthCon (1,1) (Just 64)
inCon65 = Conv fthCon (1,1) (Just 144)
inCon66 = Conv fthCon (1,1) (Just 32)
sthCon = concMod inCon61 inCon62 inCon63 inCon64

inCon71 = Conv 192 (1,1) (Just 256)
inCon72 = Conv 96 (3,3) (Just 320)
inCon73 = Conv 16 (5,5) (Just 128)
inCon74 = Conv 192 (1,1) (Just 128)
inCon75 = Conv 192 (1,1) (Just 160)
inCon76 = Conv 192 (1,1) (Just 32)
sntCon = concMod inCon71 inCon72 inCon73 inCon74

inCon81 = Conv sntCon (1,1) (Just 256)
inCon82 = Conv 160 (3,3) (Just 320)
inCon83 = Conv 32 (5,5) (Just 128)
inCon84 = Conv sntCon (1,1) (Just 128)
inCon85 = Conv sntCon (1,1) (Just 160)
inCon86 = Conv sntCon (1,1) (Just 32)
ethCon = concMod inCon81 inCon82 inCon83 inCon84

inCon91 = Conv ethCon (1,1) (Just 384)
inCon92 = Conv 192 (3,3) (Just 384)
inCon93 = Conv 48 (5,5) (Just 128)
inCon94 = Conv ethCon (1,1) (Just 128)
inCon95 = Conv ethCon (1,1) (Just 192)
inCon96 = Conv ethCon (1,1) (Just 48)
nthCon = concMod inCon91 inCon92 inCon93 inCon94

fullCon1 = FullyCon (Just nthCon) 1000

c1 = convLayer con1
c2 = convLayer con2
c3 = convLayer con3

im1 = calcInMod inCon11 inCon12 inCon13 inCon14 inCon15 inCon16
im2 = calcInMod inCon21 inCon22 inCon23 inCon24 inCon25 inCon26
im3 = calcInMod inCon31 inCon32 inCon33 inCon34 inCon35 inCon36
im4 = calcInMod inCon41 inCon42 inCon43 inCon44 inCon45 inCon46
im5 = calcInMod inCon51 inCon52 inCon53 inCon54 inCon55 inCon56
im6 = calcInMod inCon61 inCon62 inCon63 inCon64 inCon65 inCon66
im7 = calcInMod inCon71 inCon72 inCon73 inCon74 inCon75 inCon76
im8 = calcInMod inCon81 inCon82 inCon83 inCon84 inCon85 inCon86
im9 = calcInMod inCon91 inCon92 inCon93 inCon94 inCon95 inCon96

fc1 = fcLayer fullCon1

totalParamWoDrop = c1 + c2 + c3 + im1 + im2 + im3 + im4 + im5 + im6 + im7 + im8 + im9 + fc1
