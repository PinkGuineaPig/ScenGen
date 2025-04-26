from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, ARRAY
from sqlalchemy.orm import relationship, joinedload
from Backend.app import db

# ---------------------------
# ModelRunConfig: Stores all model-related hyperparameters set by the user
# ---------------------------
class ModelRunConfig(db.Model):
    __tablename__ = 'model_run_config'

    id             = Column(Integer, primary_key=True, nullable=False)
    created_at     = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    model_type     = Column(String(50), nullable=False)
    parameters     = Column(JSON,    nullable=False)
    currency_pairs = Column(ARRAY(String(10)), nullable=False, default=list)

    # Relationships
    pca_configs = relationship(
        'PCAProjectionConfig',
        back_populates='config',
        lazy='select',
        cascade='all, delete-orphan'
    )
    som_configs = relationship(
        'SOMProjectionConfig',
        back_populates='config',
        lazy='select',
        cascade='all, delete-orphan'
    )
    run = relationship(
        'ModelRun',
        uselist=False,
        back_populates='config',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return (
            f'<ModelRunConfig id={self.id} '
            f'type={self.model_type}>'
        )

    def to_dict(self, include_relations=False):
        data = {
            'id':             self.id,
            'model_type':     self.model_type,
            'parameters':     self.parameters,
            'currency_pairs': self.currency_pairs,
            'created_at':     self.created_at.isoformat(),
        }
        if include_relations:
            data['run']         = self.run.to_frontend_dict() if self.run else None
            data['pca_configs'] = [cfg.to_dict() for cfg in self.pca_configs]
            data['som_configs'] = [cfg.to_dict() for cfg in self.som_configs]
        return data

    def to_flat_dicts(self):
        rows = []
        pcas = self.pca_configs or [None]
        soms = self.som_configs or [None]
        flat_params = self.parameters.copy()

        for pca in pcas:
            for som in soms:
                row = {
                    'id':                self.id,
                    'model_type':        self.model_type,
                    **flat_params,
                    'currency_pairs':    ','.join(self.currency_pairs),
                    'created_at':        self.created_at.isoformat(),
                    'pca_n_components':  getattr(pca, 'n_components', None),
                    'pca_whiten':        (pca.additional_params or {}).get('whiten') if pca else None,
                    'pca_solver':        (pca.additional_params or {}).get('solver') if pca else None,
                    'som_dims':          f"{som.x_dim}|{som.y_dim}" if som else None,
                    'som_iterations':    getattr(som, 'iterations', None),
                    'som_sigma':         (som.additional_params or {}).get('sigma') if som else None,
                    'som_learning_rate': (som.additional_params or {}).get('learning_rate') if som else None,
                }
                rows.append(row)
        return rows

    @classmethod
    def all_flat_dicts(cls, session):
        configs = (
            session.query(cls)
                   .options(
                       joinedload(cls.pca_configs),
                       joinedload(cls.som_configs),
                       joinedload(cls.run)
                   )
                   .all()
        )
        flat = []
        for cfg in configs:
            flat.extend(cfg.to_flat_dicts())
        return flat

# ---------------------------
# ModelRun: Stores the trained model created by the user (one-to-one with config)
# ---------------------------
class ModelRun(db.Model):
    __tablename__ = 'model_run'

    id          = Column(Integer, primary_key=True, nullable=False)
    created_at  = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    config_id   = Column(
        Integer,
        ForeignKey('model_run_config.id', ondelete='CASCADE'),
        unique=True,
        nullable=False
    )
    version     = Column(Integer, nullable=False)
    model_blob  = Column(db.LargeBinary, nullable=False)

    config = relationship(
        'ModelRunConfig',
        back_populates='run'
    )
    loss_history = relationship(
        'ModelLossHistory',
        back_populates='run',
        cascade='all, delete-orphan'
    )
    latent_points = relationship(
        'LatentPoint',
        back_populates='run',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<ModelRun id={self.id} v={self.version}>'

    def to_dict(self):
        return {
            'id':         self.id,
            'version':    self.version,
            'created_at': self.created_at.isoformat(),
            'model_blob': self.model_blob,
        }

    def to_frontend_dict(self):
        return {
            'id':         self.id,
            'version':    self.version,
            'created_at': self.created_at.isoformat(),
        }

# ---------------------------
# ModelLossHistory: Records loss/error for each epoch of a model run
# ---------------------------
class ModelLossHistory(db.Model):
    __tablename__ = 'model_loss_history'

    id           = Column(Integer, primary_key=True, nullable=False)
    model_run_id = Column(Integer, ForeignKey('model_run.id', ondelete='CASCADE'), nullable=False)
    epoch        = Column(Integer, nullable=False)
    loss_type    = Column(String(50), nullable=False)
    value        = Column(Float, nullable=False)

    run = relationship(
        'ModelRun',
        back_populates='loss_history'
    )

    def __repr__(self):
        return f'<LossHistory run={self.model_run_id} epoch={self.epoch}>'

    def to_dict(self):
        return {
            'id':            self.id,
            'model_run_id':  self.model_run_id,
            'epoch':         self.epoch,
            'loss_type':     self.loss_type,
            'value':         self.value,
        }