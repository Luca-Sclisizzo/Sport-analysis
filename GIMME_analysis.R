library(gimme)

dir.create('C:/Users/lucas/OneDrive - Università degli Studi di Trieste/TRIENNALE/TESI_RICERCA/DATI/FISI/GIMME/OUT')
TS_samples = 'C:/Users/lucas/OneDrive - Università degli Studi di Trieste/TRIENNALE/TESI_RICERCA/DATI/FISI/GIMME/TEMPORAL_SERIES DATA'


fit <-gimmeSEM(data = TS_samples,
               out = 'C:/Users/lucas/OneDrive - Università degli Studi di Trieste/TRIENNALE/TESI_RICERCA/DATI/FISI/GIMME/OUT',
               sep = ",",
               ar = TRUE,
               header = TRUE,
               standardize = TRUE,
               subgroup = TRUE)

