// client/src/pages/About.jsx
import React from 'react';
import { Container, Typography, Box, Card, CardMedia, CardContent } from '@mui/material';
import { motion } from 'framer-motion';
import homeImage from '../assets/images/home_page.jpg';

const About = () => {
  return (
    <Container
      sx={{
        py: 5,
        position: 'relative',
        zIndex: 2,
        backgroundColor: 'rgba(255,255,255,0.85)',
        borderRadius: 2,
        boxShadow: 3,
        mt: 4,
      }}
    >
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
        <Typography variant="h3" align="center" gutterBottom>
          About Wine Recommender
        </Typography>
      </motion.div>

      {/* Inner box to hold card, also semi-transparent */}
      <Box
        sx={{
          mt: 4,
          p: 2,
          backgroundColor: 'rgba(255,255,255,0.75)',
          borderRadius: 2,
          boxShadow: 2,
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Card
            sx={{
              maxWidth: 700,
              backgroundColor: 'transparent',
              boxShadow: 'none',
            }}
          >
            <CardMedia
              component="img"
              height="300"
              image={homeImage}
              alt="Wine bottle"
              sx={{ borderRadius: 1 }}
            />
            <CardContent>
              <Typography variant="body1" color="text.secondary">
                Welcome to Wine Recommender, your personal guide to discovering the perfect wine for every occasion. Our AI-powered platform analyzes your taste preferences and pairs them with our extensive wine catalog to provide personalized recommendations.
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Container>
  );
};

export default About;
