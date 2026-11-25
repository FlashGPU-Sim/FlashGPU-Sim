# Native compiled cluster dimensions

Native cubin launches now retain `EIATTR_EXPLICIT_CLUSTER` and its required
`EIATTR_CTA_PER_CLUSTER` dimensions through SASSIR `CLUSTERDIMS` and runtime
resource metadata. A default launch uses those dimensions; a matching explicit
launch is accepted. The common grid validation, CTA residency and cluster
scheduler remain unchanged. Kernels without this metadata retain their existing
behavior, and the PTX path is unchanged.

The exporter and decoder reject missing, duplicate, zero or overflowing required
dimensions. The native launch adapter rejects explicit dimensions that conflict
with the compiled requirement. Preferred cluster sizes are not interpreted as
required sizes.

`sm100-compiled-cluster` tests default launch ranks, matching explicit launches
and rejection of a grid that is not divisible by the compiled dimensions, using
the same executable through PTX and SASS. Host and Python tests cover metadata
validation and isolation between kernels. This does not add new cluster special
registers or change cluster timing. A possible existing PTX discrepancy for
conflicting explicit dimensions needs separate qualification.
