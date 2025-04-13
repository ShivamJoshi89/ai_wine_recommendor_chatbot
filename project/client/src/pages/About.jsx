// client/src/pages/About.jsx
import React from 'react';
import { Container, Typography, Box, CardMedia } from '@mui/material';
import { motion } from 'framer-motion';
import homeImage from '../assets/images/home_page.jpg';

const About = () => {
  return (
    <Container
      sx={{
        position: 'relative',
        zIndex: 2,
        mt: 4,
        mb: 8,
        px: { xs: 2, md: 4 },
        backgroundColor: 'transparent',
      }}
    >
      {/* Page Title */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Typography
          variant="h3"
          align="center"
          gutterBottom
          sx={{
            color: 'text.primary',
            textTransform: 'uppercase',
            fontWeight: 700,
            letterSpacing: 2,
          }}
        >
          About Wine Recommender
        </Typography>
        <Typography
          variant="subtitle1"
          align="center"
          sx={{ color: 'text.secondary', mb: 4, fontStyle: 'italic' }}
        >
          Your personal sommelier in the digital age.
        </Typography>
      </motion.div>

      {/* Glassmorphic Panel */}
      <Box
        sx={{
          position: 'relative',
          mx: 'auto',
          maxWidth: 900,
          borderRadius: 3,
          overflow: 'hidden',
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.15)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.37)',
        }}
      >
        {/* Background Image */}
        <CardMedia
          component="div"
          sx={{
            height: { xs: 200, md: 300 },
            backgroundImage: `url(${homeImage})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            filter: 'brightness(0.6)',
          }}
        />

        {/* Text Content */}
        <Box sx={{ p: { xs: 3, md: 5 } }}>
          <Typography
            variant="body1"
            align="center"
            sx={{ color: 'text.primary', fontSize: '1.1rem', mb: 2, lineHeight: 1.6 }}
          >
            Welcome to <strong>Wine Recommender</strong>, your personal guide to discovering the perfect
            wine for every occasion. Our AI‑powered platform analyzes your taste preferences and pairs
            them with our extensive wine catalog to provide personalized recommendations.
          </Typography>
          <Typography
            variant="body2"
            align="center"
            sx={{ color: 'text.secondary', fontSize: '0.9rem', lineHeight: 1.5 }}
          >
            Whether you’re a seasoned connoisseur or just beginning your journey, we make it effortless
            to explore varietals, regions, and vintages. Cheers to discovering your next favorite bottle!
          </Typography>
        </Box>
      </Box>
    </Container>
  );
};

export default About;
