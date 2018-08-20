# option 	description
# font 	Integer specifying font to use for text.
# 1=plain, 2=bold, 3=italic, 4=bold italic, 5=symbol
# font.axis 	font for axis annotation
# font.lab 	font for x and y labels
# font.main 	font for titles
# font.sub 	font for subtitles
# ps 	font point size (roughly 1/72 inch)
# text size=ps*cex
# family 	font family for drawing text. Standard values are "serif", "sans", "mono", "symbol". Mapping is device dependent.
# 
# In windows, mono is mapped to "TT Courier New", serif is mapped to"TT Times New Roman", sans is mapped to "TT Arial", mono is mapped to "TT Courier New", and symbol is mapped to "TT Symbol" (TT=True Type). You can add your own mappings. 

par(font.main=1, font.lab=1, font.sub=1,family = "serif")

mapes <- structure(c(12.3726814731951,
11.235445803361,
18.0913473936859,
10.9280619076006,
14.2696246050274,
13.7355487364372,
12.4328354705247,
16.4795129261239,
10.8425889072139,
13.7455453075378,
12.9682946754931,
11.1182099067703,
16.8114538916213,
11.0395914805514,
14.1290785957816,
13.5332998747361,
11.1041953259017,
22.9986490660064,
12.1956348579277,
14.153548097216,
13.0259006093929,
11.0335668469393,
18.7271404918167,
10.8501419024291,
14.7432051061444), .Dim = c(5L, 5L), .Dimnames = list(
c("NIC", "QLD", "SA", "TAS", "VIC"), NULL))
set.seed(1)
xx <- runif(nrow(mapes)*ncol(mapes),-0.3,0.3)+rep(1:ncol(mapes),nrow(mapes))
plot(xx,as.vector(mapes),pch=19,xaxt="n",ylab="",xlab="",main="MAPE with GD (5 Fold split)")
axis(1,seq_along(rownames(mapes)),rownames(mapes))


mapes <- structure(c(12.4078142054279,
11.1334648143781,
18.1127909756667,
10.9570671978126,
14.3590650663282,
12.393716288086,
11.2020834438718,
16.5592116289425,
10.840589331739,
13.7141239095969,
13.0056108831111,
11.1167532202091,
16.748811562466,
10.9864675255704,
14.162280658406,
12.758487191289,
11.0775945297984,
21.9834697758068,
10.6848459218082,
14.1757375131975,
12.9854220212321,
11.0187585109404,
18.6901876288465,
10.8086233949363,
14.7600835261074), .Dim = c(5L, 5L), .Dimnames = list(
c("NIC", "QLD", "SA", "TAS", "VIC"), NULL))
set.seed(1)
xx <- runif(nrow(mapes)*ncol(mapes),-0.3,0.3)+rep(1:ncol(mapes),nrow(mapes))
plot(xx,as.vector(mapes),pch=19,xaxt="n",ylab="",xlab="",main="MAPE with CSO (5 Fold split)")
axis(1,seq_along(rownames(mapes)),rownames(mapes))


mapes <- structure(c(14.642697974025,
11.7248235509452,
30.759817371827,
10.886361584371,
15.187314517862,
13.0559760904246,
11.1311463067603,
20.6625029598177,
10.8207080521198,
14.0744827975153,
12.4462837256148,
11.2864488574802,
17.4707416817986,
11.4163998283825,
14.2995579495271,
13.2388326563369,
11.5895622251059,
19.088055697368,
10.8831711343261,
14.4484425543129,
12.4922652654486,
11.0098522085725,
18.0685601594875,
10.7851018253672,
13.6294060064126), .Dim = c(5L, 5L), .Dimnames = list(
c("NIC", "QLD", "SA", "TAS", "VIC"), NULL))
set.seed(1)
xx <- runif(nrow(mapes)*ncol(mapes),-0.3,0.3)+rep(1:ncol(mapes),nrow(mapes))
plot(xx,as.vector(mapes),pch=19,xaxt="n",ylab="",xlab="",main="MAPE with GD (Timeseries split)")
axis(1,seq_along(rownames(mapes)),rownames(mapes))


mapes <- structure(c(13.2086662010617,
10.9277384868669,
18.6489080502233,
10.184373449826,
14.7493516027612,
12.7661011478904,
11.2521739274651,
17.1506309576127,
10.8929914483245,
13.7823900776995,
12.3311446638586,
10.9914081577064,
17.8889881439833,
10.7359454884711,
13.9719083864594,
12.4198888133992,
11.2437045843862,
17.3317678071419,
10.9505525230517,
14.5959372447541,
13.1093847520303,
11.090287412159,
18.0264941037663,
10.3666995580342,
14.3688753885113), .Dim = c(5L, 5L), .Dimnames = list(
c("NIC", "QLD", "SA", "TAS", "VIC"), NULL))
set.seed(1)
xx <- runif(nrow(mapes)*ncol(mapes),-0.3,0.3)+rep(1:ncol(mapes),nrow(mapes))
plot(xx,as.vector(mapes),pch=19,xaxt="n",ylab="",xlab="",main="MAPE with CSO (Timeseries split)")
axis(1,seq_along(rownames(mapes)),rownames(mapes))

