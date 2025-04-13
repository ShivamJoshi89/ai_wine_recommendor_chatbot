// client/src/pages/Register.jsx
import React, { useState } from 'react';
import { Container, Typography, Box, TextField, Button, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { register } from '../services/authService';

const Register = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail]     = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]     = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await register({ username, email, password });
      navigate('/login');
    } catch (err) {
      console.error('Register error:', err);
      setError('Registration failed. Please try again.');
    }
  };

  return (
    <Container
      maxWidth="sm"
      sx={{
        position: 'relative',
        zIndex: 2,
        mt: 8,
        mb: 8,
        px: 2,
        backgroundColor: 'transparent',
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Typography
          variant="h4"
          align="center"
          gutterBottom
          sx={{
            color: 'text.primary',
            textTransform: 'uppercase',
            letterSpacing: 2,
            fontWeight: 700,
          }}
        >
          Register
        </Typography>
      </motion.div>

      <Paper
        elevation={0}
        sx={{
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: 3,
          boxShadow: '0 8px 32px rgba(0,0,0,0.37)',
          p: 4,
          mt: 2,
        }}
      >
        <Box component="form" onSubmit={handleSubmit} noValidate>
          <TextField
            label="Username"
            variant="filled"
            fullWidth
            margin="normal"
            InputProps={{
              sx: {
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'text.primary',
                '& .MuiInputBase-input': { color: 'text.primary' },
                '& .MuiInputLabel-root': { color: 'text.secondary' },
              },
            }}
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />

          <TextField
            label="Email"
            variant="filled"
            fullWidth
            margin="normal"
            type="email"
            InputProps={{
              sx: {
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'text.primary',
                '& .MuiInputBase-input': { color: 'text.primary' },
                '& .MuiInputLabel-root': { color: 'text.secondary' },
              },
            }}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <TextField
            label="Password"
            variant="filled"
            fullWidth
            margin="normal"
            type="password"
            InputProps={{
              sx: {
                backgroundColor: 'rgba(255,255,255,0.1)',
                color: 'text.primary',
                '& .MuiInputBase-input': { color: 'text.primary' },
                '& .MuiInputLabel-root': { color: 'text.secondary' },
              },
            }}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          {error && (
            <Typography variant="body2" color="error" sx={{ mt: 1 }}>
              {error}
            </Typography>
          )}

          <Button
            type="submit"
            variant="contained"
            fullWidth
            sx={{ mt: 3, textTransform: 'uppercase', py: 1.5 }}
          >
            Register
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default Register;
