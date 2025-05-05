from Backend.app import db
from datetime import datetime

# ---------------------------
# LatentPoint: Stores latent-space vectors for each input sequence snapshot.
# ---------------------------
class LatentPoint(db.Model):
    __tablename__ = 'latent_point'

    id              = db.Column(db.Integer, primary_key=True)
    # Changed from Date to DateTime (UTC)
    start_date      = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True
    )
    lag             = db.Column(db.Integer, nullable=False)
    latent_vector   = db.Column(db.ARRAY(db.Float), nullable=True)
    model_run_id    = db.Column(
        db.Integer,
        db.ForeignKey('model_run.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    __table_args__ = (
        db.UniqueConstraint(
            'model_run_id',
            'start_date',
            'lag',
            name='uq_latent_seq'
        ),
    )

    run = db.relationship(
        'ModelRun',
        back_populates='latent_points'
    )

    pca_projections = db.relationship(
        'PCAProjection',
        back_populates='point',
        cascade='all, delete-orphan'
    )

    som_projections = db.relationship(
        'SOMProjection',
        back_populates='point',
        cascade='all, delete-orphan'
    )

    inputs = db.relationship(
        'LatentInput',
        back_populates='point',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        ts = self.start_date.isoformat() if self.start_date else None
        return f"<LatentPoint id={self.id} start_date={ts} lag={self.lag}>"

    def get_pca_vector(self, config_id=None):
        projs = self.pca_projections
        if config_id is not None:
            projs = [p for p in projs if p.config_id == config_id]
        return [p.value for p in sorted(projs, key=lambda p: p.dim)]

    def get_som_coordinates(self, session, config_id=None):
        query = session.query(SOMProjection).filter_by(latent_point_id=self.id)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return query.first()

# ---------------------------
# LatentInput: Links each latent point back to its original data source.
# ---------------------------
class LatentInput(db.Model):
    __tablename__ = 'latent_input'

    id              = db.Column(db.Integer, primary_key=True)
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id', ondelete='CASCADE'), nullable=False, index=True)
    currency_pair   = db.Column(db.String(10), nullable=False)
    date            = db.Column(db.Date, nullable=False)
    source_table    = db.Column(db.String(80), nullable=False)

    point = db.relationship('LatentPoint', back_populates='inputs')

    def __repr__(self):
        return f"<LatentInput point={self.latent_point_id} pair={self.currency_pair} date={self.date}>"

# ---------------------------
# SOMProjectionConfig: Stores configuration details for SOM projections.
# ---------------------------
class SOMProjectionConfig(db.Model):
    __tablename__ = 'som_projection_config'

    id              = db.Column(db.Integer, primary_key=True)
    model_config_id = db.Column(
        db.Integer,
        db.ForeignKey('model_run_config.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    x_dim           = db.Column(db.Integer, nullable=False)
    y_dim           = db.Column(db.Integer, nullable=False)
    iterations      = db.Column(db.Integer, nullable=False)
    additional_params = db.Column(db.JSON, nullable=True,default=dict)
    created_at      = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)

    config = db.relationship(
        'ModelRunConfig',
        back_populates='som_configs'
    )

    projections = db.relationship(
        'SOMProjection',
        backref=db.backref('som_config', lazy='joined'),
        lazy='select',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return (
            f'<SOMConfig model_config={self.model_config_id} '
            f'{self.x_dim}x{self.y_dim} iters={self.iterations}>'
        )

    def to_dict(self):
        return {
            'id': self.id,
            'model_config_id': self.model_config_id,
            'x_dim': self.x_dim,
            'y_dim': self.y_dim,
            'iterations': self.iterations,
            'additional_params': self.additional_params,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ---------------------------
# SOMProjection: Maps latent points onto a 2D SOM grid cell.
# ---------------------------
class SOMProjection(db.Model):
    __tablename__ = 'som_projection'

    id              = db.Column(db.Integer, primary_key=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id', ondelete='CASCADE'), nullable=False, index=True)
    config_id       = db.Column(db.Integer, db.ForeignKey('som_projection_config.id', ondelete='CASCADE'), nullable=False, index=True)
    x               = db.Column(db.Integer, nullable=False)
    y               = db.Column(db.Integer, nullable=False)

    config = db.relationship(
        'SOMProjectionConfig',
        back_populates='projections'
    )

    point = db.relationship(
        'LatentPoint',
        back_populates='som_projections'
    )

    @classmethod
    def get_latent_points_for_cell(cls, session, x, y, config_id=None):
        query = session.query(cls).filter_by(x=x, y=y)
        if config_id is not None:
            query = query.filter_by(config_id=config_id)
        return [p.point for p in query.all()]

# ---------------------------
# PCAProjectionConfig: Stores configuration details for PCA projections.
# ---------------------------
class PCAProjectionConfig(db.Model):
    __tablename__ = 'pca_projection_config'

    id              = db.Column(db.Integer, primary_key=True)
    model_config_id = db.Column(
        db.Integer,
        db.ForeignKey('model_run_config.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    n_components    = db.Column(db.Integer, nullable=False)
    additional_params = db.Column(db.JSON, nullable=True)
    explained_variance = db.Column(db.JSON, nullable=True)
    components         = db.Column(db.JSON,    nullable=True)
    created_at      = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)

    config = db.relationship(
        'ModelRunConfig',
        back_populates='pca_configs'
    )

    projections = db.relationship(
        'PCAProjection',
        back_populates='config',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f'<PCAConfig model_config={self.model_config_id} components={self.n_components}>'

    def to_dict(self):
        return {
            'id': self.id,
            'model_config_id': self.model_config_id,
            'n_components': self.n_components,
            'additional_params': self.additional_params,
            'explained_variance':  self.explained_variance,
            'components':         self.components,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# ---------------------------
# PCAProjection: Stores individual PCA coordinate values for latent points.
# ---------------------------
class PCAProjection(db.Model):
    __tablename__ = 'pca_projection'

    id              = db.Column(db.Integer, primary_key=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    latent_point_id = db.Column(db.Integer, db.ForeignKey('latent_point.id', ondelete='CASCADE'), nullable=False, index=True)
    config_id       = db.Column(db.Integer, db.ForeignKey('pca_projection_config.id', ondelete='CASCADE'), nullable=False, index=True)
    dim             = db.Column(db.Integer, nullable=False)
    value           = db.Column(db.Float, nullable=False)

    config = db.relationship(
        'PCAProjectionConfig',
        back_populates='projections'
    )

    point = db.relationship(
        'LatentPoint',
        back_populates='pca_projections'
    )

    @classmethod
    def get_vectors_for_config(cls, session, config_id):
        query = session.query(cls).filter_by(config_id=config_id)
        results = query.order_by(cls.latent_point_id, cls.dim).all()
        vectors = {}
        for p in results:
            vectors.setdefault(p.latent_point_id, []).append(p.value)
        return vectors
