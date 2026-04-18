from models.mamba_selective_copying import SelectiveCopyingMambaModel


class InductionHeadsMambaModel(SelectiveCopyingMambaModel):
    """
    Token-level LM wrapper for the Induction Heads task.

    Currently identical to SelectiveCopyingMambaModel, but kept as a separate
    class so the two experiments can diverge (different pooling, different
    metrics, ...) without re-plumbing the pipeline.
    """
    pass
