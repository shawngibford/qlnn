# Quantum-Liquid Neural ODE for Time Series Generation
## Complete Implementation Plan for Bioreactor OD Data Augmentation

**Project Goal**: Generate realistic synthetic time series sequences of optical density (OD) and related bioreactor features to augment a small dataset (778 samples) for improved model training.

**Key Innovation**: Neural Architecture Search for quantum circuits combined with Liquid Neural ODEs, optimized for Apple Silicon M1/MPS.

---

## 📊 Dataset Summary

- **Size**: 778 samples × 9 features over 5.4 days
- **Sampling**: 10-minute intervals (mostly regular)
- **Target**: Optical Density (OD) - range [0.47, 3.80]
- **Features**: PRE, TEMP_EXT, TEMP_CULTURE, PAR_LIGHT, PH, DO, OD, DRY, CELL
- **Characteristics**: 
  - High autocorrelation (0.95+ at 8hr lag)
  - Non-stationary with strong trend (R²=0.75)
  - Three growth phases: lag, log, stationary
  - Clean data, no missing values

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTUM-LIQUID NEURAL ODE                     │
└─────────────────────────────────────────────────────────────────┘

Input Sequence (T×6)
    ↓
┌─────────────────────────────┐
│  Quantum Feature Encoder    │  ← Automatically searched architecture
│  - Variational Circuit      │
│  - Data Re-uploading        │
│  - Parameterized Gates      │
└─────────────────────────────┘
    ↓ (quantum expectations)
Classical Latent Vector (D_latent)
    ↓
┌─────────────────────────────┐
│  Liquid Neural ODE Cell     │
│  - Continuous-time dynamics │
│  - Adaptive ODE solver      │
│  - Learnable time constants │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Decoder Network            │
│  - MLP to output space      │
│  - Multi-output (6 features)│
└─────────────────────────────┘
    ↓
Generated Sequence (T×6)
```

---

## 🔬 Phase 1: Classical Baseline - Liquid Neural ODE

### Architecture Specifications

**Liquid Cell Configuration:**
- **Neuron type**: Continuous-Time RNN (CT-RNN)
- **State dimension**: 32 neurons
- **ODE formulation**: 
  ```
  dx/dt = -x/τ + f(Wx·x + Wu·u + b)
  ```
  where:
  - x: hidden state
  - τ: learnable time constants (one per neuron)
  - f: tanh activation
  - u: input at time t

**ODE Solver:**
- Method: Dormand-Prince (dopri5) with adaptive stepping
- Tolerance: rtol=1e-3, atol=1e-4
- Integration: torchdiffeq library (MPS compatible)

**Network Structure:**
- Input: 6 features (excluding DRY, CELL, OD - we predict OD)
- Encoder: Linear(6 → 32)
- Liquid ODE Cell: 32 → 32 (continuous)
- Decoder: Linear(32 → 6) with skip connections
- Output: [PRE, TEMP_EXT, TEMP_CULTURE, PAR_LIGHT, PH, DO]

**Physics-Informed Constraints:**
1. **Growth model regularization**: Add logistic growth prior
   ```
   L_physics = λ * |dOD/dt - μ·OD·(1 - OD/K)|²
   ```
2. **Smoothness constraint**: Penalize unrealistic jumps
   ```
   L_smooth = β * |∇²OD|²
   ```
3. **Non-negativity**: All physical quantities ≥ 0

### Loss Function

```python
L_total = L_reconstruction + λ_physics·L_physics + λ_smooth·L_smooth + λ_kl·L_KL
```

where:
- L_reconstruction: MSE between real and generated sequences
- L_KL: KL divergence for variational regularization (optional)
- Weights: λ_physics=0.1, λ_smooth=0.05

---

## ⚛️ Phase 2: Quantum Feature Encoder with Architecture Search

### Quantum Circuit Neural Architecture Search (QNAS)

**Search Space:**
- **Qubits**: 4-8 (start with 4-6 for M1 efficiency)
- **Depth**: 2-6 layers
- **Gates**: {RX, RY, RZ, CNOT, CZ, CRX, CRY}
- **Entanglement patterns**: Linear, all-to-all, circular, custom
- **Measurement basis**: Pauli-Z, Pauli-X, Pauli-Y, or combinations
- **Data encoding**: Amplitude, angle, basis encoding, or hybrid

**Search Strategy (Evolution-based):**

```python
class QuantumCircuitSearcher:
    """
    Evolutionary search for optimal quantum circuit architecture.
    Similar to RL but uses genetic algorithms for discrete space.
    """
    
    def __init__(self, n_qubits_range=(4, 6), depth_range=(2, 6)):
        self.population_size = 20
        self.n_generations = 50
        self.mutation_rate = 0.2
        self.crossover_rate = 0.6
        
    def generate_random_circuit(self):
        """Generate random circuit from search space"""
        n_qubits = random.randint(*self.n_qubits_range)
        depth = random.randint(*self.depth_range)
        
        circuit = []
        for layer in range(depth):
            # Rotation gates
            gate_type = random.choice(['RX', 'RY', 'RZ'])
            qubit_idx = random.randint(0, n_qubits-1)
            circuit.append((gate_type, qubit_idx, layer))
            
            # Entanglement
            if random.random() < 0.7:  # 70% chance of entanglement
                entangle_type = random.choice(['CNOT', 'CZ'])
                control = random.randint(0, n_qubits-1)
                target = random.randint(0, n_qubits-1)
                if control != target:
                    circuit.append((entangle_type, control, target, layer))
        
        return {
            'n_qubits': n_qubits,
            'depth': depth,
            'gates': circuit,
            'encoding': random.choice(['angle', 'amplitude', 'hybrid'])
        }
    
    def fitness_function(self, circuit, data_loader):
        """
        Evaluate circuit quality based on:
        1. Reconstruction error
        2. Feature expressivity (quantum fisher information)
        3. Circuit complexity penalty
        4. Training speed
        """
        model = build_model_with_circuit(circuit)
        
        # Quick evaluation (5 epochs)
        loss = train_quick(model, data_loader, epochs=5)
        
        # Expressivity measure
        qfi = compute_quantum_fisher_information(circuit)
        
        # Complexity penalty
        n_params = count_parameters(circuit)
        complexity_penalty = 0.01 * n_params
        
        # Fitness (minimize)
        fitness = loss - 0.1*qfi + complexity_penalty
        return fitness
    
    def mutate(self, circuit):
        """Randomly modify circuit structure"""
        if random.random() < self.mutation_rate:
            mutation_type = random.choice([
                'add_gate', 'remove_gate', 'change_gate', 
                'add_entanglement', 'change_qubit_count'
            ])
            # Apply mutation...
        return circuit
    
    def crossover(self, parent1, parent2):
        """Combine two circuits"""
        # Take gates from both parents
        child_gates = []
        for g1, g2 in zip(parent1['gates'], parent2['gates']):
            child_gates.append(g1 if random.random() < 0.5 else g2)
        
        child = {
            'n_qubits': random.choice([parent1['n_qubits'], parent2['n_qubits']]),
            'depth': random.choice([parent1['depth'], parent2['depth']]),
            'gates': child_gates,
            'encoding': random.choice([parent1['encoding'], parent2['encoding']])
        }
        return child
    
    def search(self, data_loader, save_path):
        """Run evolutionary search"""
        # Initialize population
        population = [self.generate_random_circuit() 
                     for _ in range(self.population_size)]
        
        best_circuit = None
        best_fitness = float('inf')
        
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness_scores = []
            for circuit in population:
                fitness = self.fitness_function(circuit, data_loader)
                fitness_scores.append((circuit, fitness))
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1])
            
            # Update best
            if fitness_scores[0][1] < best_fitness:
                best_fitness = fitness_scores[0][1]
                best_circuit = fitness_scores[0][0]
                print(f"Gen {generation}: New best fitness = {best_fitness:.4f}")
            
            # Selection (top 50%)
            survivors = [c for c, f in fitness_scores[:self.population_size//2]]
            
            # Create next generation
            next_population = survivors.copy()
            
            # Crossover
            while len(next_population) < self.population_size:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)
            
            population = next_population
            
            # Save checkpoint
            if generation % 10 == 0:
                save_checkpoint(best_circuit, generation, save_path)
        
        return best_circuit
```

**Quantum Feature Encoder Structure (after search):**

```python
class QuantumFeatureEncoder(nn.Module):
    """
    Encode classical features into quantum states and measure.
    Architecture determined by QNAS.
    """
    
    def __init__(self, circuit_config, n_features=6):
        super().__init__()
        self.n_qubits = circuit_config['n_qubits']
        self.circuit = build_pennylane_circuit(circuit_config)
        
        # Learnable parameters for quantum gates
        self.n_params = count_circuit_params(circuit_config)
        self.quantum_weights = nn.Parameter(
            torch.randn(self.n_params) * 0.1
        )
        
        # Classical pre-processing
        self.feature_embed = nn.Linear(n_features, self.n_qubits * 2)
        
        # Classical post-processing
        self.latent_dim = 16
        self.post_process = nn.Linear(self.n_qubits, self.latent_dim)
    
    def forward(self, x):
        """
        x: (batch, sequence, features)
        returns: (batch, sequence, latent_dim)
        """
        batch, seq, feats = x.shape
        
        # Reshape for processing
        x_flat = x.reshape(batch * seq, feats)
        
        # Embed to quantum input dimension
        x_embed = self.feature_embed(x_flat)
        
        # Apply quantum circuit
        quantum_out = self.quantum_circuit(x_embed, self.quantum_weights)
        
        # Post-process measurements
        latent = self.post_process(quantum_out)
        
        # Reshape back
        latent = latent.reshape(batch, seq, self.latent_dim)
        
        return latent
    
    @qml.qnode(qml.device('default.qubit', wires=n_qubits))
    def quantum_circuit(self, inputs, weights):
        """
        PennyLane quantum circuit (structure from QNAS).
        Implement data re-uploading for expressivity.
        """
        # Data encoding (depends on QNAS result)
        # Example: Angle encoding with re-uploading
        
        weight_idx = 0
        for layer in range(depth):
            # Encode data
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Parameterized layer
            for gate in layer_gates[layer]:
                if gate['type'] == 'RX':
                    qml.RX(weights[weight_idx], wires=gate['qubit'])
                    weight_idx += 1
                elif gate['type'] == 'CNOT':
                    qml.CNOT(wires=[gate['control'], gate['target']])
                # ... other gates
            
            # Re-upload data
            for i in range(self.n_qubits):
                qml.RX(inputs[i + self.n_qubits], wires=i)
        
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
```

---

## 🔧 Phase 3: Complete Model Integration

### Full Model Architecture

```python
class QuantumLiquidNeuralODE(nn.Module):
    """
    Complete model: Quantum encoder → Liquid ODE → Decoder
    Optimized for Apple M1/MPS
    """
    
    def __init__(self, 
                 quantum_circuit_config,
                 n_features=6,
                 liquid_size=32,
                 output_size=6):
        super().__init__()
        
        # Quantum encoder
        self.quantum_encoder = QuantumFeatureEncoder(
            quantum_circuit_config, 
            n_features
        )
        
        # Liquid ODE cell
        self.liquid_cell = LiquidCell(
            input_size=self.quantum_encoder.latent_dim,
            hidden_size=liquid_size,
            ode_solver='dopri5'
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(liquid_size, 64),
            nn.Tanh(),
            nn.Linear(64, output_size)
        )
        
        # Physics-informed components
        self.growth_model = LogisticGrowthModel()
        
    def forward(self, x, time_points):
        """
        x: (batch, sequence, features)
        time_points: (sequence,) - actual timestamps in hours
        """
        # Quantum encoding
        z = self.quantum_encoder(x)  # (batch, seq, latent)
        
        # Liquid ODE dynamics
        # Integrate over time using ODE solver
        h0 = z[:, 0, :]  # Initial hidden state
        
        # Solve ODE: dh/dt = liquid_cell(h, z, t)
        def ode_func(t, h):
            # Interpolate input z at time t
            t_idx = self._find_time_index(t, time_points)
            z_t = self._interpolate_z(z, t_idx)
            return self.liquid_cell(h, z_t, t)
        
        # Adaptive integration
        solution = odeint(
            ode_func, 
            h0, 
            time_points,
            method='dopri5',
            rtol=1e-3,
            atol=1e-4
        )  # (sequence, batch, liquid_size)
        
        # Transpose for decoding
        h_seq = solution.transpose(0, 1)  # (batch, sequence, liquid_size)
        
        # Decode to output space
        output = self.decoder(h_seq)  # (batch, sequence, output_size)
        
        return output
    
    def generate_sequence(self, initial_condition, n_steps, dt=10/60):
        """
        Generate new synthetic sequence.
        
        initial_condition: (batch, features) - starting point
        n_steps: number of time steps to generate
        dt: time step in hours (default 10min)
        """
        self.eval()
        with torch.no_grad():
            batch_size = initial_condition.shape[0]
            
            # Initialize
            current_state = initial_condition.unsqueeze(1)  # (batch, 1, features)
            generated = [current_state]
            
            # Encode initial state
            z = self.quantum_encoder(current_state)
            h = z.squeeze(1)  # (batch, latent)
            
            time_points = torch.arange(n_steps) * dt
            
            for step in range(1, n_steps):
                t = time_points[step]
                
                # Evolve hidden state
                h = self._single_ode_step(h, z, t, dt)
                
                # Decode
                output = self.decoder(h)  # (batch, features)
                
                # Store
                generated.append(output.unsqueeze(1))
                
                # Update z for next step (re-encode)
                z = self.quantum_encoder(output.unsqueeze(1))
                z = z.squeeze(1)
            
            # Concatenate
            generated_seq = torch.cat(generated, dim=1)
            
        return generated_seq
```

### Liquid Cell Implementation

```python
class LiquidCell(nn.Module):
    """
    Continuous-time RNN cell with learnable time constants.
    Based on Liquid Time-Constant Networks (LTC).
    """
    
    def __init__(self, input_size, hidden_size, ode_solver='dopri5'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Learnable time constants (positive via softplus)
        self.tau = nn.Parameter(torch.randn(hidden_size) * 0.5)
        
        # Synaptic weights
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_x = nn.Linear(input_size, hidden_size)
        
        # Activation
        self.activation = nn.Tanh()
        
    def forward(self, h, x, t):
        """
        Compute dh/dt for ODE solver.
        
        h: (batch, hidden_size) - current hidden state
        x: (batch, input_size) - current input
        t: scalar - current time
        """
        # Time constants (ensure positive)
        tau = F.softplus(self.tau) + 0.1  # Add epsilon for stability
        
        # Compute update
        pre_activation = self.W_h(h) + self.W_x(x)
        f_h = self.activation(pre_activation)
        
        # Continuous-time dynamics: dh/dt = -h/τ + f(Wh·h + Wx·x)
        dh_dt = (-h / tau) + f_h
        
        return dh_dt
```

---

## 📈 Data Preprocessing & Augmentation

### Feature Engineering

```python
class BioreactorDataPreprocessor:
    """
    Prepare bioreactor data for training.
    """
    
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], dayfirst=True, format='mixed')
        
        # Remove redundant features (DRY, CELL perfectly correlated with OD)
        self.feature_cols = ['PRE', 'TEMP_EXT', 'TEMP_CULTURE', 
                            'PAR_LIGHT', 'PH', 'DO']
        self.target_col = 'OD'
        
    def normalize(self):
        """Min-max normalization to [0, 1] for quantum circuits"""
        self.scalers = {}
        
        for col in self.feature_cols + [self.target_col]:
            scaler = MinMaxScaler()
            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler
        
        return self
    
    def create_sequences(self, window_size=24, stride=1):
        """
        Create sliding window sequences.
        
        window_size: number of timesteps per sequence (24 = 4 hours)
        stride: step between windows (1 = 10 minutes)
        """
        sequences = []
        targets = []
        
        features = self.df[self.feature_cols].values
        target = self.df[self.target_col].values
        
        for i in range(0, len(self.df) - window_size, stride):
            seq = features[i:i+window_size]
            tgt = target[i:i+window_size]
            
            sequences.append(seq)
            targets.append(tgt)
        
        sequences = np.array(sequences)  # (n_samples, window_size, n_features)
        targets = np.array(targets)      # (n_samples, window_size)
        
        print(f"Created {len(sequences)} sequences from {len(self.df)} samples")
        print(f"Effective dataset size multiplier: {len(sequences)/len(self.df):.1f}x")
        
        return sequences, targets
    
    def augment_data(self, sequences, targets, augmentation_factor=5):
        """
        Apply time series augmentation techniques.
        """
        augmented_seqs = [sequences]
        augmented_tgts = [targets]
        
        for _ in range(augmentation_factor - 1):
            # Time warping (stretch/compress by ±10%)
            warped_seqs = self._time_warp(sequences, sigma=0.1)
            augmented_seqs.append(warped_seqs)
            augmented_tgts.append(targets)  # targets stay same
            
            # Magnitude warping (scale by ±5%)
            mag_warped = self._magnitude_warp(sequences, sigma=0.05)
            augmented_seqs.append(mag_warped)
            augmented_tgts.append(targets)
            
            # Jittering (add small noise)
            jittered = self._jitter(sequences, sigma=0.02)
            augmented_seqs.append(jittered)
            augmented_tgts.append(targets)
        
        all_seqs = np.concatenate(augmented_seqs, axis=0)
        all_tgts = np.concatenate(augmented_tgts, axis=0)
        
        print(f"Augmented dataset: {len(all_seqs)} sequences")
        
        return all_seqs, all_tgts
    
    def _time_warp(self, sequences, sigma=0.1):
        """Warp time axis by random smoothed curve"""
        warped = []
        for seq in sequences:
            # Generate smooth random curve
            warp = np.cumsum(np.random.randn(len(seq)) * sigma)
            warp = (warp - warp.min()) / (warp.max() - warp.min()) * len(seq)
            
            # Interpolate
            orig_steps = np.arange(len(seq))
            warped_seq = np.zeros_like(seq)
            for feat in range(seq.shape[1]):
                warped_seq[:, feat] = np.interp(warp, orig_steps, seq[:, feat])
            
            warped.append(warped_seq)
        
        return np.array(warped)
    
    def _magnitude_warp(self, sequences, sigma=0.05):
        """Scale magnitude by smooth random curve"""
        warped = []
        for seq in sequences:
            # Smooth scaling curve
            scales = 1 + np.random.randn(len(seq)) * sigma
            scales = np.convolve(scales, np.ones(5)/5, mode='same')
            
            warped_seq = seq * scales[:, np.newaxis]
            warped.append(warped_seq)
        
        return np.array(warped)
    
    def _jitter(self, sequences, sigma=0.02):
        """Add Gaussian noise"""
        noise = np.random.randn(*sequences.shape) * sigma
        return sequences + noise
```

### Train/Val/Test Split (Temporal)

```python
def temporal_split(sequences, targets, train_ratio=0.65, val_ratio=0.15):
    """
    Split temporally (no shuffling!) to test future generalization.
    """
    n_samples = len(sequences)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_seq = sequences[:train_end]
    train_tgt = targets[:train_end]
    
    val_seq = sequences[train_end:val_end]
    val_tgt = targets[train_end:val_end]
    
    test_seq = sequences[val_end:]
    test_tgt = targets[val_end:]
    
    print(f"Split: Train={len(train_seq)}, Val={len(val_seq)}, Test={len(test_seq)}")
    
    return (train_seq, train_tgt), (val_seq, val_tgt), (test_seq, test_tgt)
```

---

## 🚀 Training Protocol

### Training Loop with MPS Optimization

```python
class Trainer:
    """
    Training manager with MPS optimization for M1.
    """
    
    def __init__(self, model, device='mps'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer (AdamW for better generalization)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss weights
        self.lambda_physics = 0.1
        self.lambda_smooth = 0.05
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'physics_loss': [],
            'smooth_loss': []
        }
    
    def physics_informed_loss(self, pred_od, time_points):
        """
        Regularize with logistic growth model.
        
        Logistic model: dOD/dt = μ·OD·(1 - OD/K)
        where μ is growth rate, K is carrying capacity
        """
        # Estimate derivatives
        dt = time_points[1] - time_points[0]
        dod_dt = torch.diff(pred_od, dim=1) / dt
        
        # Logistic growth expectation
        mu = 0.3  # Estimated from data
        K = 3.8   # Max OD observed
        
        od_mid = pred_od[:, :-1]  # Match derivative length
        expected_dod_dt = mu * od_mid * (1 - od_mid / K)
        
        # L2 loss
        loss = F.mse_loss(dod_dt, expected_dod_dt)
        
        return loss
    
    def smoothness_loss(self, pred_sequence):
        """
        Penalize unrealistic jumps (second derivative).
        """
        # Second-order differences
        first_diff = torch.diff(pred_sequence, dim=1)
        second_diff = torch.diff(first_diff, dim=1)
        
        # L2 norm
        loss = torch.mean(second_diff ** 2)
        
        return loss
    
    def train_epoch(self, train_loader, time_points):
        """Single training epoch"""
        self.model.train()
        total_loss = 0
        total_physics = 0
        total_smooth = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            pred = self.model(x, time_points)
            
            # Reconstruction loss
            loss_recon = F.mse_loss(pred, x)  # Reconstruct input features
            
            # Physics-informed loss (on OD prediction)
            # Assume OD is first output dimension (or extract appropriately)
            pred_od = pred[:, :, 0]  # Adjust index based on your feature order
            true_od = y
            
            loss_physics = self.physics_informed_loss(pred_od, time_points)
            loss_smooth = self.smoothness_loss(pred_od)
            
            # Total loss
            loss = (loss_recon + 
                   self.lambda_physics * loss_physics + 
                   self.lambda_smooth * loss_smooth)
            
            # Backward pass with gradient clipping for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track
            total_loss += loss.item()
            total_physics += loss_physics.item()
            total_smooth += loss_smooth.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_physics = total_physics / len(train_loader)
        avg_smooth = total_smooth / len(train_loader)
        
        return avg_loss, avg_physics, avg_smooth
    
    def validate(self, val_loader, time_points):
        """Validation pass"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                
                pred = self.model(x, time_points)
                loss = F.mse_loss(pred, x)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(self, train_loader, val_loader, time_points, epochs=100, 
            save_path='checkpoints/best_model.pt'):
        """
        Full training loop with early stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Train
            train_loss, physics_loss, smooth_loss = self.train_epoch(
                train_loader, time_points
            )
            
            # Validate
            val_loss = self.validate(val_loader, time_points)
            
            # Track
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['physics_loss'].append(physics_loss)
            self.history['smooth_loss'].append(smooth_loss)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Physics: {physics_loss:.6f}, Smooth: {smooth_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, save_path)
                print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Load best model
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
        
        return self.history
```

### MPS-Specific Optimizations

```python
# MPS optimization settings for M1
def setup_mps_device():
    """
    Configure PyTorch for optimal M1/MPS performance.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ MPS (Metal Performance Shaders) available")
        
        # MPS-specific settings
        torch.backends.mps.enabled = True
        
        # Set optimal thread count for M1
        torch.set_num_threads(8)  # M1 has 8 cores
        
        return device
    else:
        print("MPS not available, using CPU")
        return torch.device("cpu")

# Mixed precision for faster training (if supported)
def use_mixed_precision():
    """
    Enable mixed precision training for M1.
    Note: MPS support for autocast is limited, monitor compatibility.
    """
    # Check compatibility
    try:
        scaler = torch.cuda.amp.GradScaler()  # May work with MPS
        return True
    except:
        print("Mixed precision not available, using FP32")
        return False

# Efficient data loading for M1
def create_dataloader(sequences, targets, batch_size=32, shuffle=True):
    """
    Create optimized DataLoader for M1.
    """
    dataset = TensorDataset(
        torch.from_numpy(sequences).float(),
        torch.from_numpy(targets).float()
    )
    
    # M1 optimization: pin_memory=False (MPS doesn't use it)
    # num_workers=0 (MPS works best with main thread)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=True  # For consistent batch sizes
    )
    
    return loader
```

---

## 📊 Evaluation Framework

### Metrics for Generated Sequences

```python
class SequenceEvaluator:
    """
    Comprehensive evaluation of generated time series.
    """
    
    def __init__(self, scaler):
        self.scaler = scaler  # For denormalization
    
    def evaluate_all(self, real_sequences, generated_sequences):
        """
        Compute all evaluation metrics.
        """
        results = {}
        
        # 1. Reconstruction metrics
        results['mse'] = self.mse(real_sequences, generated_sequences)
        results['mae'] = self.mae(real_sequences, generated_sequences)
        results['r2'] = self.r2_score(real_sequences, generated_sequences)
        
        # 2. Statistical similarity
        results['dtw_distance'] = self.dynamic_time_warping(
            real_sequences, generated_sequences
        )
        results['distribution_similarity'] = self.distribution_distance(
            real_sequences, generated_sequences
        )
        
        # 3. Temporal properties
        results['autocorr_similarity'] = self.autocorrelation_similarity(
            real_sequences, generated_sequences
        )
        results['spectral_similarity'] = self.spectral_distance(
            real_sequences, generated_sequences
        )
        
        # 4. Physics-based validation
        results['growth_rate_realism'] = self.validate_growth_rates(
            generated_sequences
        )
        results['range_validity'] = self.check_valid_ranges(
            generated_sequences
        )
        
        return results
    
    def mse(self, real, generated):
        """Mean Squared Error"""
        return np.mean((real - generated) ** 2)
    
    def mae(self, real, generated):
        """Mean Absolute Error"""
        return np.mean(np.abs(real - generated))
    
    def r2_score(self, real, generated):
        """R² coefficient"""
        ss_res = np.sum((real - generated) ** 2)
        ss_tot = np.sum((real - real.mean()) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def dynamic_time_warping(self, real, generated, max_samples=100):
        """
        DTW distance (computationally expensive, sample subset).
        """
        from dtaidistance import dtw
        
        # Sample random pairs
        n_samples = min(max_samples, len(real))
        indices = np.random.choice(len(real), n_samples, replace=False)
        
        distances = []
        for idx in indices:
            # Compute DTW for each feature
            for feat in range(real.shape[2]):
                dist = dtw.distance(
                    real[idx, :, feat],
                    generated[idx, :, feat]
                )
                distances.append(dist)
        
        return np.mean(distances)
    
    def distribution_distance(self, real, generated):
        """
        Wasserstein distance between distributions.
        """
        from scipy.stats import wasserstein_distance
        
        distances = []
        for feat in range(real.shape[2]):
            real_flat = real[:, :, feat].flatten()
            gen_flat = generated[:, :, feat].flatten()
            
            dist = wasserstein_distance(real_flat, gen_flat)
            distances.append(dist)
        
        return np.mean(distances)
    
    def autocorrelation_similarity(self, real, generated, max_lag=24):
        """
        Compare autocorrelation functions.
        """
        def compute_acf(sequences, max_lag):
            acf = np.zeros((sequences.shape[2], max_lag))
            for feat in range(sequences.shape[2]):
                data = sequences[:, :, feat].flatten()
                for lag in range(max_lag):
                    if lag == 0:
                        acf[feat, lag] = 1.0
                    else:
                        acf[feat, lag] = np.corrcoef(
                            data[:-lag], data[lag:]
                        )[0, 1]
            return acf
        
        real_acf = compute_acf(real, max_lag)
        gen_acf = compute_acf(generated, max_lag)
        
        # L2 distance
        similarity = np.mean((real_acf - gen_acf) ** 2)
        
        return similarity
    
    def spectral_distance(self, real, generated):
        """
        Compare power spectral densities.
        """
        from scipy.signal import welch
        
        distances = []
        for feat in range(real.shape[2]):
            real_flat = real[:, :, feat].flatten()
            gen_flat = generated[:, :, feat].flatten()
            
            # Compute PSDs
            f_real, psd_real = welch(real_flat, nperseg=256)
            f_gen, psd_gen = welch(gen_flat, nperseg=256)
            
            # L2 distance
            dist = np.sqrt(np.mean((psd_real - psd_gen) ** 2))
            distances.append(dist)
        
        return np.mean(distances)
    
    def validate_growth_rates(self, generated):
        """
        Check if growth rates are realistic.
        """
        # Denormalize OD (assume first feature)
        od = generated[:, :, 0]
        od_denorm = self.scaler.inverse_transform(
            od.reshape(-1, 1)
        ).reshape(od.shape)
        
        # Compute growth rates
        growth_rates = np.diff(od_denorm, axis=1)
        
        # Check if within realistic bounds (from real data analysis)
        # Real data: growth rate ~ [-0.5, 0.5] OD/timestep
        realistic = np.mean(
            (growth_rates >= -0.5) & (growth_rates <= 0.5)
        )
        
        return realistic * 100  # Percentage
    
    def check_valid_ranges(self, generated):
        """
        Check if values are within physically valid ranges.
        """
        # Define valid ranges (normalized to [0, 1])
        valid_ranges = {
            0: (0.0, 1.0),  # OD
            1: (0.0, 1.0),  # PH
            2: (0.0, 1.0),  # DO
            3: (0.0, 1.0),  # TEMP
            # ... etc
        }
        
        violations = 0
        total = 0
        
        for feat, (min_val, max_val) in valid_ranges.items():
            feat_data = generated[:, :, feat]
            violations += np.sum((feat_data < min_val) | (feat_data > max_val))
            total += feat_data.size
        
        validity = (1 - violations / total) * 100
        
        return validity
```

### Visualization

```python
def visualize_results(real_seq, generated_seq, save_path='results/'):
    """
    Create comprehensive visualization comparing real and generated sequences.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    feature_names = ['OD', 'PH', 'DO', 'TEMP_CULTURE', 'TEMP_EXT', 'PAR_LIGHT']
    
    for idx, (ax, feat_name) in enumerate(zip(axes.flat, feature_names)):
        # Plot several example sequences
        n_examples = 5
        for i in range(n_examples):
            # Real
            ax.plot(real_seq[i, :, idx], alpha=0.6, 
                   color='blue', linestyle='-', linewidth=1)
            # Generated
            ax.plot(generated_seq[i, :, idx], alpha=0.6, 
                   color='red', linestyle='--', linewidth=1)
        
        ax.set_title(f'{feat_name} - Real (blue) vs Generated (red)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Normalized Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comparison.png', dpi=150)
    
    # Distribution comparison
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, (ax, feat_name) in enumerate(zip(axes2.flat, feature_names)):
        real_flat = real_seq[:, :, idx].flatten()
        gen_flat = generated_seq[:, :, idx].flatten()
        
        ax.hist(real_flat, bins=50, alpha=0.5, label='Real', color='blue')
        ax.hist(gen_flat, bins=50, alpha=0.5, label='Generated', color='red')
        
        ax.set_title(f'{feat_name} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/distributions.png', dpi=150)
```

---

## 📁 Complete Code Structure

```
quantum_liquid_neuralode/
│
├── data/
│   ├── raw/
│   │   └── data.csv
│   ├── processed/
│   │   ├── sequences_train.npy
│   │   ├── sequences_val.npy
│   │   └── sequences_test.npy
│   └── augmented/
│       └── augmented_sequences.npy
│
├── models/
│   ├── __init__.py
│   ├── quantum_encoder.py          # Quantum feature encoder
│   ├── liquid_cell.py               # Liquid ODE cell
│   ├── full_model.py                # Complete architecture
│   └── physics_constraints.py       # Physics-informed components
│
├── qnas/                            # Quantum Neural Architecture Search
│   ├── __init__.py
│   ├── search_space.py              # Define circuit search space
│   ├── evolutionary_search.py       # Evolution algorithm
│   ├── fitness_evaluator.py         # Circuit evaluation
│   └── circuit_builder.py           # Build PennyLane circuits
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                   # Training loop
│   ├── losses.py                    # Loss functions
│   └── mps_utils.py                 # M1 optimization utilities
│
├── data_processing/
│   ├── __init__.py
│   ├── preprocessor.py              # Data cleaning & normalization
│   ├── augmentation.py              # Time series augmentation
│   └── sequence_generator.py        # Create sequences from raw data
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # Evaluation metrics
│   └── visualizations.py            # Plotting functions
│
├── utils/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   └── logger.py                    # Logging utilities
│
├── checkpoints/                     # Saved models
│   ├── best_model.pt
│   ├── best_circuit.json
│   └── qnas_results/
│
├── results/                         # Outputs
│   ├── figures/
│   ├── generated_sequences/
│   └── evaluation_reports/
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_qnas_search.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_generation_evaluation.ipynb
│
├── scripts/                         # Standalone scripts
│   ├── run_qnas.py                  # Execute quantum circuit search
│   ├── train_model.py               # Train full model
│   ├── generate_sequences.py        # Generate synthetic data
│   └── evaluate_model.py            # Run full evaluation
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── config.yaml                      # Configuration file
```

---

## 🔧 Dependencies & Setup

### requirements.txt

```txt
# Core ML frameworks
torch>=2.0.0
torchdiffeq>=0.2.3

# Quantum computing
pennylane>=0.32.0
pennylane-lightning>=0.32.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Time series utilities
dtaidistance>=2.3.10
scipy>=1.11.0

# Optimization
optuna>=3.3.0  # Optional: for hyperparameter tuning

# Utils
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0

# Development
jupyter>=1.0.0
ipython>=8.14.0
```

### Installation Instructions

```bash
# Create virtual environment
python -m venv venv_qlnode
source venv_qlnode/bin/activate  # On Mac

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## ⚙️ Configuration (config.yaml)

```yaml
# Data settings
data:
  path: "data/raw/data.csv"
  window_size: 24  # 4 hours at 10-min intervals
  stride: 6        # 1 hour overlap
  augmentation_factor: 5
  features:
    - PRE
    - TEMP_EXT
    - TEMP_CULTURE
    - PAR_LIGHT
    - PH
    - DO
  target: OD

# Quantum circuit search
qnas:
  n_qubits_range: [4, 6]
  depth_range: [2, 6]
  population_size: 20
  n_generations: 50
  mutation_rate: 0.2
  crossover_rate: 0.6
  gates: [RX, RY, RZ, CNOT, CZ]
  encodings: [angle, amplitude, hybrid]

# Model architecture
model:
  quantum_latent_dim: 16
  liquid_size: 32
  n_features: 6
  output_size: 6

# Training
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  patience: 20
  
  # Loss weights
  lambda_physics: 0.1
  lambda_smooth: 0.05
  
  # Optimizer
  optimizer: AdamW
  scheduler: ReduceLROnPlateau
  
  # MPS settings
  device: mps
  num_threads: 8

# ODE solver
ode:
  method: dopri5
  rtol: 0.001
  atol: 0.0001

# Physics constraints
physics:
  growth_model: logistic
  mu: 0.3  # Growth rate
  K: 3.8   # Carrying capacity

# Evaluation
evaluation:
  n_generated_sequences: 1000
  metrics:
    - mse
    - mae
    - r2
    - dtw
    - distribution_similarity
    - autocorrelation_similarity
    - spectral_similarity

# Paths
paths:
  checkpoints: "checkpoints/"
  results: "results/"
  logs: "logs/"
```

---

## 🚀 Execution Pipeline

### Step 1: Data Preparation

```bash
python scripts/prepare_data.py --config config.yaml
```

This will:
- Load and clean data
- Create normalized sequences
- Apply augmentation
- Save train/val/test splits

### Step 2: Quantum Circuit Search

```bash
python scripts/run_qnas.py --config config.yaml --output checkpoints/best_circuit.json
```

Expected output:
- `best_circuit.json`: Optimized quantum circuit architecture
- `qnas_results/`: Fitness evolution plots, population history

### Step 3: Train Full Model

```bash
python scripts/train_model.py \
    --config config.yaml \
    --circuit checkpoints/best_circuit.json \
    --output checkpoints/best_model.pt
```

Expected training time (M1 Mac):
- QNAS: 2-4 hours
- Full training: 30-60 minutes (100 epochs)

### Step 4: Generate Synthetic Sequences

```bash
python scripts/generate_sequences.py \
    --model checkpoints/best_model.pt \
    --n_sequences 1000 \
    --output results/generated_sequences.npy
```

### Step 5: Evaluate Results

```bash
python scripts/evaluate_model.py \
    --real data/processed/sequences_test.npy \
    --generated results/generated_sequences.npy \
    --output results/evaluation_report.json
```

---

## 📊 Expected Performance

Based on dataset characteristics and architecture:

**Classical Baseline (Phase 1):**
- Reconstruction R²: 0.85 - 0.90
- Training time: ~20 minutes
- Parameters: ~15K

**Quantum-Enhanced (Phase 2):**
- Reconstruction R²: 0.88 - 0.93
- Training time: ~40 minutes
- Parameters: ~8K (50% reduction)
- Sample efficiency: 2-3x improvement

**Full Model with Physics (Phase 3):**
- Reconstruction R²: 0.90 - 0.95
- DTW distance: < 0.15
- Distribution similarity: > 0.90
- Valid growth rates: > 95%

---

## 🔬 Key Mathematical Formulations

### Liquid ODE Dynamics

The continuous-time recurrent cell evolves according to:

$$\frac{dx}{dt} = -\frac{x}{\tau} + \tanh(W_h x + W_u u + b)$$

where:
- $x \in \mathbb{R}^{n}$: hidden state
- $\tau \in \mathbb{R}^{n}_+$: learnable time constants
- $u \in \mathbb{R}^{m}$: input
- $W_h \in \mathbb{R}^{n \times n}$, $W_u \in \mathbb{R}^{n \times m}$: weight matrices

### Quantum Feature Encoding

For a feature vector $f \in \mathbb{R}^d$, the quantum encoding is:

$$|\psi(f; \theta)\rangle = U_L(\theta_L) \cdots U_1(\theta_1) \prod_{i=1}^{n} R_Y(f_i) |0\rangle^{\otimes n}$$

where:
- $U_l(\theta_l)$: parameterized unitary layer
- $R_Y(f_i)$: rotation encoding of feature $i$
- $n$: number of qubits

Measurement expectations:
$$z_i = \langle \psi | \sigma_z^{(i)} | \psi \rangle$$

### Physics-Informed Loss

Total loss combines reconstruction and physical constraints:

$$\mathcal{L} = \mathcal{L}_{recon} + \lambda_{phys} \mathcal{L}_{phys} + \lambda_{smooth} \mathcal{L}_{smooth}$$

where:

$$\mathcal{L}_{phys} = \mathbb{E}\left[\left|\frac{d\text{OD}}{dt} - \mu \cdot \text{OD} \cdot \left(1 - \frac{\text{OD}}{K}\right)\right|^2\right]$$

$$\mathcal{L}_{smooth} = \mathbb{E}\left[\left|\frac{d^2\text{OD}}{dt^2}\right|^2\right]$$

---

## 🎯 Success Criteria

The model is considered successful if:

1. **Reconstruction Accuracy**: R² > 0.90 on test set
2. **Distribution Fidelity**: Wasserstein distance < 0.1
3. **Temporal Coherence**: Autocorrelation similarity > 0.95
4. **Physical Validity**: 
   - Growth rates within realistic bounds (>95%)
   - No negative values
   - Smooth trajectories (no discontinuities)
5. **Sample Efficiency**: Achieves performance with <50% of parameters vs classical baseline

---

## 🔍 Debugging & Troubleshooting

### Common Issues

**Issue 1: MPS not available**
```python
# Check MPS availability
import torch
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True
```
Solution: Update PyTorch to latest version with MPS support

**Issue 2: ODE solver divergence**
- Reduce tolerance: rtol=1e-4, atol=1e-5
- Decrease learning rate
- Add gradient clipping (already included)
- Check initial conditions aren't extreme

**Issue 3: Quantum circuit too slow**
- Reduce number of qubits (start with 4)
- Decrease circuit depth
- Use lightning.qubit instead of default.qubit in PennyLane

**Issue 4: Poor generation quality**
- Increase augmentation factor
- Adjust physics loss weight
- Ensure proper temporal splitting (no data leakage)
- Check feature normalization

### Performance Optimization

```python
# Profile code to find bottlenecks
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your training code here
trainer.fit(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

---

## 📝 Next Steps After Implementation

1. **Hyperparameter Tuning**: Use Optuna for automated search
2. **Ensemble Models**: Train multiple models with different initializations
3. **Transfer Learning**: Fine-tune on related bioreactor datasets
4. **Production Deployment**: Export to ONNX for inference optimization
5. **Uncertainty Quantification**: Add Bayesian layers for confidence estimates

---

## 📚 References & Further Reading

**Liquid Neural Networks:**
- Hasani et al. (2021). "Liquid Time-Constant Networks". AAAI.
- Lechner et al. (2020). "Neural Circuit Policies". Nature Machine Intelligence.

**Quantum Machine Learning:**
- Schuld et al. (2020). "Circuit-centric quantum classifiers". Physical Review A.
- Benedetti et al. (2019). "Parameterized quantum circuits as ML models". Quantum Science and Technology.

**Neural Architecture Search:**
- Liu et al. (2019). "DARTS: Differentiable Architecture Search". ICLR.
- Real et al. (2019). "Regularized Evolution for Image Classifier Architecture Search". AAAI.

**Physics-Informed Neural Networks:**
- Raissi et al. (2019). "Physics-informed neural networks". Journal of Computational Physics.
- Karniadakis et al. (2021). "Physics-informed machine learning". Nature Reviews Physics.

---

## 💡 Final Notes

This implementation plan provides a complete roadmap for building a quantum-enhanced liquid neural ODE for time series generation. The approach is specifically tailored for:

✓ Small datasets (778 samples)
✓ Bioreactor/biological time series
✓ Apple Silicon M1 optimization
✓ Quantum circuit architecture search
✓ Physics-informed constraints

The code structure is modular and extensible, allowing for easy experimentation and modification. All components are designed to work together seamlessly while maintaining clear separation of concerns.

**Key Advantages:**
1. Automatic quantum circuit optimization (no manual design needed)
2. Continuous-time modeling (handles irregular sampling naturally)
3. Physics-informed regularization (improves generalization)
4. Efficient computation on M1 (MPS-optimized)
5. Comprehensive evaluation framework

Good luck with the implementation! The architecture search phase will be particularly interesting given your quantum circuits experience. The evolutionary approach should find some creative circuit designs that outperform hand-crafted ones.

