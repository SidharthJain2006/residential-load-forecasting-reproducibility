# Dataset rights and provenance review — required before public release

## UCI
The UCI Individual Household Electric Power Consumption dataset record lists CC BY 4.0. Attribute the dataset to Georges Hebrail and Alice Berard and cite DOI 10.24432/C58K54. The raw data file is not bundled; the corrected processed windows are included for this private staging build.

## MORED
The official MORED page provides downloads and describes the dataset as open, but the inspected landing page did not state a clear data licence. The processed V3 files are included in this **private staging bundle** to preserve the tested split. Do not make the MORED-derived values publicly accessible until the authors confirm the applicable redistribution terms with the dataset provider. The original APT6.csv file is not bundled.

## London
The split CSVs and results in this bundle are associated with the Low Carbon London/UK Power Networks data. The London Datastore record lists a Creative Commons Attribution licence. Before public release, confirm that the files were derived from that open UKPN/GLA release. Do not distribute UK Data Service Study 7857 data under its separate End User Licence; that is a distinct access route with registered-user restrictions.

## Required action
The corresponding author should confirm exact source archive/version and applicable licence for MORED and London, and add any required attribution and licence notices. If rights cannot be confirmed, remove the affected derived data files and replace them with download instructions and deterministic preparation code.
