// client/src/pages/Profile.jsx
import React from 'react';
import { Container, Typography, Box, Paper, Avatar, Button } from '@mui/material';
import { motion } from 'framer-motion';

const Profile = () => {
  // Example user data
  const user = {
    name: 'John Doe',
    email: 'johndoe@example.com',
    avatar: 'https://source.unsplash.com/random/100x100/?portrait'
  };

  return (
    <Container maxWidth="sm" sx={{ py: 5 }}>
      <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }}>
        <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
          <Avatar alt={user.name} src={user.avatar} sx={{ width: 100, height: 100, margin: '0 auto' }} />
          <Typography variant="h5" sx={{ mt: 2 }}>
            {user.name}
          </Typography>
          <Typography variant="body1" color="text.secondary">
            {user.email}
          </Typography>
          <Box sx={{ mt: 3 }}>
            <Button variant="contained">Edit Profile</Button>
          </Box>
        </Paper>
      </motion.div>
    </Container>
  );
};

export default Profile;
