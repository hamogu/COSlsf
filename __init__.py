import COSlsf

for n in COSlsf.lsf_names:

    # Add to module level namespace
    globals()[n] = getattr(COSlsf, n)

# keep module namespace clean
del n
del COSlsf
