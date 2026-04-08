################################################################################
###### Brain Plot using Freesurfer Atlas
###### Zixuan Wen
###### Date: 251111
################################################################################

rm(list = ls())
library(ggseg)
library(dplyr)
library(data.table)
library(ggplot2)
library(patchwork)
library(RColorBrewer)

# Check freesurfer names, further manully align with my data
data(dk);    sort(unique(dk$data$region))
data(aseg);  sort(unique(aseg$data$region))

setwd("YOUR_Directory")


## load data
df_plot <- as.data.frame(fread("YOUR_ROI_DATA_ALL.csv",
                               sep = ",",header = T,stringsAsFactors = F))

# Inputs:
# mri_df_cortical   : data.frame(region=<DK name>, value=<numeric>)
# mri_df_subcortical: data.frame(region=<ASEG name>, value=<numeric>)
data(dk)    # cortical
data(aseg)  # subcortical

mri_df_cortical <- df_plot[df_plot$dk==1, c("ggseg_dk", "mri")]
mri_df_subcortical <- df_plot[df_plot$dk==0, c("ggseg_dk", "mri")]
colnames(mri_df_cortical) <- c("region", "value")
colnames(mri_df_subcortical) <- c("region", "value")

## Shared limits across cortical + subcortical so the scale is identical ---
vmin <- min(c(mri_df_cortical$value, mri_df_subcortical$value), na.rm = TRUE)
vmax <- max(c(mri_df_cortical$value, mri_df_subcortical$value), na.rm = TRUE)

## Helper to subset atlas by hemi/side safely
subset_atlas <- function(atlas, hemi = NULL, side = NULL){
  at <- atlas
  dd <- at$data
  if(!is.null(hemi) && "hemi" %in% names(dd)) dd <- dd[dd$hemi %in% hemi, , drop = FALSE]
  if(!is.null(side) && "side" %in% names(dd)) dd <- dd[dd$side %in% side, , drop = FALSE]
  at$data <- dd
  at
}


## Cortical: Left-Lateral & Left-Medial (DK)
dk_L_lat <- subset_atlas(dk,  hemi = "left", side = "lateral")
dk_L_med <- subset_atlas(dk,  hemi = "left", side = "medial")

## define a reusable fill scale with shared limits
fill_mri <- scale_fill_gradient(low = "lightblue", high = "darkblue",
                                name = "MRI", limits = c(vmin, vmax))

p_LL <- ggplot(mri_df_cortical) +
  geom_brain(atlas = dk_L_lat, data = mri_df_cortical,
             mapping = aes(region = region, fill = value)) +
  fill_mri +
  theme_void() +
  theme(legend.position = "left")  # legend will be collected

p_LM <- ggplot(mri_df_cortical) +
  geom_brain(atlas = dk_L_med, data = mri_df_cortical,
             mapping = aes(region = region, fill = value)) +
  fill_mri +
  theme_void() +
  theme(legend.position = "left")

## Subcortical: Coronal & Sagittal (ASEG)
aseg_cor <- subset_atlas(aseg, side = "coronal")
aseg_sag <- subset_atlas(aseg, side = "sagittal")

p_SC <- ggplot(mri_df_subcortical) +
  geom_brain(atlas = aseg_cor, data = mri_df_subcortical,
             mapping = aes(region = region, fill = value)) +
  fill_mri +
  theme_void() +
  theme(legend.position = "left")

p_SS <- ggplot(mri_df_subcortical) +
  geom_brain(atlas = aseg_sag, data = mri_df_subcortical,
             mapping = aes(region = region, fill = value)) +
  fill_mri +
  theme_void() +
  theme(legend.position = "left")

##  2×2 and shared legend on the left
final_2x2 <- (p_LL | p_LM) / (p_SC | p_SS) +
  plot_layout(guides = "collect") & theme(legend.position = "left")

## display
final_2x2

final_2x2_big <- final_2x2 &
  theme(
    legend.position = "left",          # keep on left
    legend.title = element_text(size = 20, face = "bold"),
    legend.text  = element_text(size = 15),
    legend.key.height = unit(3.0, "cm"),  # make bar longer (vertical)
    legend.key.width  = unit(1, "cm")   # make bar thicker (horizontal)
  )

## save figure
ggsave(
  "YOUR_OUTPUT1.pdf",
  plot = final_2x2_big, width = 8, height = 8, dpi = 300, bg = "white"
)

################################################################################
#### similarly for amyloid
################################################################################

amy_df_cortical <- df_plot[df_plot$dk==1, c("ggseg_dk", "amy")]
amy_df_subcortical <- df_plot[df_plot$dk==0, c("ggseg_dk", "amy")]
colnames(amy_df_cortical) <- c("region", "value")
colnames(amy_df_subcortical) <- c("region", "value")


vmin <- min(c(amy_df_cortical$value, amy_df_subcortical$value), na.rm = TRUE)
vmax <- max(c(amy_df_cortical$value, amy_df_subcortical$value), na.rm = TRUE)


subset_atlas <- function(atlas, hemi = NULL, side = NULL){
  at <- atlas
  dd <- at$data
  if(!is.null(hemi) && "hemi" %in% names(dd)) dd <- dd[dd$hemi %in% hemi, , drop = FALSE]
  if(!is.null(side) && "side" %in% names(dd)) dd <- dd[dd$side %in% side, , drop = FALSE]
  at$data <- dd
  at
}


## Cortical: Left-Lateral & Left-Medial (DK)
dk_L_lat <- subset_atlas(dk,  hemi = "left", side = "lateral")
dk_L_med <- subset_atlas(dk,  hemi = "left", side = "medial")


fill_amy <- scale_fill_gradient(
  low = "lightgreen", high = "darkgreen",
  name = "PET", limits = c(vmin, vmax)
)

## Cortical 
p_LL <- ggplot(amy_df_cortical) +
  geom_brain(atlas = dk_L_lat, data = amy_df_cortical,
             mapping = aes(region = region, fill = value)) +
  fill_amy +
  theme_void() +
  theme(legend.position = "left")

p_LM <- ggplot(amy_df_cortical) +
  geom_brain(atlas = dk_L_med, data = amy_df_cortical,
             mapping = aes(region = region, fill = value)) +
  fill_amy +
  theme_void() +
  theme(legend.position = "left")

## Subcortical 
aseg_cor <- subset_atlas(aseg, side = "coronal")
aseg_sag <- subset_atlas(aseg, side = "sagittal")

p_SC <- ggplot(amy_df_subcortical) +
  geom_brain(atlas = aseg_cor, data = amy_df_subcortical,
             mapping = aes(region = region, fill = value)) +
  fill_amy +
  theme_void() +
  theme(legend.position = "left")

p_SS <- ggplot(amy_df_subcortical) +
  geom_brain(atlas = aseg_sag, data = amy_df_subcortical,
             mapping = aes(region = region, fill = value)) +
  fill_amy +
  theme_void() +
  theme(legend.position = "left")

## 2x2
final_2x2 <- (p_LL | p_LM) / (p_SC | p_SS) +
  plot_layout(guides = "collect") & theme(legend.position = "left")

## display
final_2x2

final_2x2_big <- final_2x2 &
  theme(
    legend.position = "left",          # keep on left
    legend.title = element_text(size = 20, face = "bold"),
    legend.text  = element_text(size = 15),
    legend.key.height = unit(3.0, "cm"),  # make bar longer (vertical)
    legend.key.width  = unit(1, "cm")   # make bar thicker (horizontal)
  )

## save figure
ggsave(
  "YOUR_OUTPUT2.pdf",
  plot = final_2x2_big, width = 8, height = 8, dpi = 300, bg = "white"
)
