
library(lixoftConnectors)
initializeLixoftConnectors(software = "monolix")
prjPath = "C:/Users/zy/Documents/Academic/NYU_CART/Automation/m_model.mlxtran"
loadProject(prjPath)

obstype <- c("continuous", "continuous")
names(obstype) <- c(1, 2)
hdrtype <- c("ID", "TIME", "OBSERVATION", "OBSID", "REGRESSOR")

do_the_mlx <- function(dataset_file, output_file) {
    setData(dataFile = dataset_file,
        headerTypes = hdrtype,
        observationTypes = obstype
    )
    runScenario()
    out_data_frame <- getEstimatedIndividualParameters()
    res <- out_data_frame$conditionalMode
    print(paste0("saving to ", output_file))
    write.csv(res, file = output_file, quote = FALSE, row.names=FALSE)
}

dataset_dir <- "DataSets/"
dataset_names <- list.files(path = dataset_dir,  pattern = ".*\\.csv")
output_dir <- "Outputs/"


dataset_s <- NULL
output_s <- NULL
for (name in dataset_names) {
    dataset_s <- c(dataset_s, paste0(dataset_dir,name))
    spl_name <- strsplit(name, "\\.")
    root_name <- paste0(spl_name[[1]][1:(length(spl_name)-1)])
    output_name <- paste0(root_name,"_out.txt")
    output_s <- c(output_s, paste0(output_dir, output_name))
}

for (k1 in 1:length(dataset_s)) {
    prpt<-paste0("Now processing: ", dataset_names[[k1]])
    print(prpt)
    do_the_mlx(dataset_s[[k1]], output_s[[k1]])
}