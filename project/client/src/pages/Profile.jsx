// client/src/pages/Profile.jsx
import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Avatar,
  Button,
  Grid,
  Card,
  CardMedia,
  CardContent,
  CardActions,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import { Favorite, Edit, ShoppingCart, Star } from '@mui/icons-material';
import { motion } from 'framer-motion';
import { getUserProfile, getRecommendations } from '../services/userService';

const Profile = () => {
  const [user, setUser] = useState(null);
  const [recs, setRecs] = useState([]);

  useEffect(() => {
    // Fetch user data & recommendations
    async function fetchData() {
      const profile = await getUserProfile();
      setUser(profile);
      const recommendations = await getRecommendations(profile.id);
      setRecs(recommendations);
    }
    fetchData();
  }, []);

  if (!user) return null; // or a loader

  return (
    <Container
      maxWidth="md"
      sx={{ py: 8, position: 'relative', zIndex: 2, backgroundColor: 'transparent' }}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <Paper
          elevation={0}
          sx={{
            backdropFilter: 'blur(10px)',
            backgroundColor: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: 3,
            p: 4,
          }}
        >
          {/* Header */}
          <Box sx={{ textAlign: 'center' }}>
            <Avatar
              alt={user.name}
              src={user.avatarUrl}
              sx={{ width: 120, height: 120, margin: '0 auto' }}
            />
            <Typography
              variant="h4"
              sx={{ mt: 2, textTransform: 'uppercase', letterSpacing: 2, color: 'text.primary' }}
            >
              {user.name}
            </Typography>
            <Typography variant="body1" color="text.secondary">
              {user.email}
            </Typography>
            <Button
              variant="outlined"
              startIcon={<Edit />}
              sx={{ mt: 2, textTransform: 'uppercase' }}
              onClick={() => {/* open edit dialog */}}
            >
              Edit Profile
            </Button>
          </Box>

          <Divider sx={{ my: 4 }} />

          <Grid container spacing={4}>
            {/* Favorites */}
            <Grid item xs={12} md={6}>
              <Typography variant="h6" sx={{ mb: 2, letterSpacing: 1 }}>
                <Favorite fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
                Favorite Wines
              </Typography>
              <List>
                {user.favorites.map((wine) => (
                  <ListItem key={wine.id} sx={{ py: 1 }}>
                    <ListItemText
                      primary={wine.name}
                      secondary={`${wine.region} • ${wine.vintage}`}
                    />
                    <IconButton edge="end">
                      <Star color="warning" />
                    </IconButton>
                  </ListItem>
                ))}
              </List>
            </Grid>

            {/* Purchase History */}
            <Grid item xs={12} md={6}>
              <Typography variant="h6" sx={{ mb: 2, letterSpacing: 1 }}>
                <ShoppingCart fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
                Purchase History
              </Typography>
              <List>
                {user.purchases.map((order) => (
                  <ListItem key={order.id} sx={{ py: 1 }}>
                    <ListItemText
                      primary={order.wineName}
                      secondary={`$${order.price.toFixed(2)} — ${new Date(
                        order.date
                      ).toLocaleDateString()}`}
                    />
                  </ListItem>
                ))}
              </List>
            </Grid>

            {/* Tasting Notes */}
            <Grid item xs={12}>
              <Typography variant="h6" sx={{ mb: 2, letterSpacing: 1 }}>
                Tasting Notes
              </Typography>
              {user.notes.length ? (
                user.notes.map((note) => (
                  <Paper
                    key={note.id}
                    sx={{
                      p: 2,
                      mb: 2,
                      backgroundColor: 'rgba(255,255,255,0.03)',
                      borderRadius: 2,
                    }}
                  >
                    <Typography variant="subtitle2" color="text.secondary">
                      {new Date(note.date).toLocaleDateString()}
                    </Typography>
                    <Typography variant="body1">{note.text}</Typography>
                  </Paper>
                ))
              ) : (
                <Typography variant="body2" color="text.secondary">
                  You haven’t added any tasting notes yet.
                </Typography>
              )}
            </Grid>

            {/* Recommendations */}
            <Grid item xs={12}>
              <Typography variant="h6" sx={{ mb: 2, letterSpacing: 1 }}>
                Recommended for You
              </Typography>
              <Grid container spacing={2}>
                {recs.map((wine) => (
                  <Grid item xs={12} sm={6} md={4} key={wine.id}>
                    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <CardMedia
                        component="img"
                        height="140"
                        image={wine.imageUrl}
                        alt={wine.name}
                      />
                      <CardContent sx={{ flexGrow: 1 }}>
                        <Typography gutterBottom variant="subtitle1">
                          {wine.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {wine.region} • {wine.vintage}
                        </Typography>
                      </CardContent>
                      <CardActions>
                        <Button size="small">View</Button>
                        <Button size="small">Add to Cart</Button>
                      </CardActions>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Grid>
          </Grid>
        </Paper>
      </motion.div>
    </Container>
  );
};

export default Profile;
