// client/src/pages/Wines.jsx
import React, { useEffect, useState, useCallback } from 'react';
import {
  Container,
  Typography,
  Card,
  CardMedia,
  CardContent,
  Box,
  Paper,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Slider,
  Autocomplete,
  TextField,
  Button,
  IconButton,
  Drawer,
  Divider,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Pagination from '@mui/material/Pagination';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { getWines, getWinesCount } from '../services/wineService';

const DEFAULT_PRICE_RANGE = [0, 1000];
const wineTypes = ['Red','White','Sparkling','Rosé','Dessert','Fortified'];
const regionsList = ['Bordeaux','Bourgogne','Napa Valley','Piemonte','Rhone Valley','Toscana'];
const countriesList = ['Argentina','Australia','Austria','Chile','France','Germany','Italy','Portugal','Spain','United States'];
const stylesList = ['Argentinian Malbec','Californian Cabernet Sauvignon','Central Italy Red','Spanish Red','Spanish Rioja Red'];
const pairingsList = [
  'Aperitif','Appetizers and snacks','Beef','Blue cheese','Cured Meat',
  'Fruity desserts','Game (deer, venison)','Goat\'s Milk Cheese','Lamb',
  'Lean fish','Mature and hard cheese','Mild and soft cheese','Mushrooms',
  'Pasta','Pork','Poultry','Rich fish (salmon, tuna etc)','Shellfish',
  'Spicy food','Sweet desserts','Veal','Vegetarian'
];

export default function Wines() {
  const [wines, setWines] = useState([]);
  const [page, setPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Filters
  const [selectedTypes, setSelectedTypes] = useState([]);
  const [priceRange, setPriceRange] = useState(DEFAULT_PRICE_RANGE);
  const [minRating, setMinRating] = useState(0);
  const [selectedGrapes, setSelectedGrapes] = useState([]);
  const [selectedRegions, setSelectedRegions] = useState([]);
  const [selectedCountries, setSelectedCountries] = useState([]);
  const [selectedStyles, setSelectedStyles] = useState([]);
  const [selectedPairings, setSelectedPairings] = useState([]);

  // Mobile drawer
  const [drawerOpen, setDrawerOpen] = useState(false);

  const buildFilters = () => {
    const f = {};
    if (selectedTypes.length)     f.types      = selectedTypes.join(',');
    if (priceRange[0] !== DEFAULT_PRICE_RANGE[0]) f.min_price  = priceRange[0];
    if (priceRange[1] !== DEFAULT_PRICE_RANGE[1]) f.max_price  = priceRange[1];
    if (minRating !== 0)          f.min_rating = minRating;
    if (selectedGrapes.length)    f.grapes     = selectedGrapes.join(',');
    if (selectedRegions.length)   f.regions    = selectedRegions.join(',');
    if (selectedCountries.length) f.countries  = selectedCountries.join(',');
    if (selectedStyles.length)    f.styles     = selectedStyles.join(',');
    if (selectedPairings.length)  f.pairings   = selectedPairings.join(',');
    return f;
  };

  const fetchCount = useCallback(filters => {
    getWinesCount(filters)
      .then(data => setTotalCount(data.count))
      .catch(() => setTotalCount(0));
  }, []);

  const fetchWines = useCallback((filters, pageNum) => {
    setLoading(true);
    getWines({ ...filters, limit: 20, page: pageNum })
      .then(data => setWines(data))
      .catch(() => setWines([]))
      .finally(() => setLoading(false));
  }, []);

  // Initial load
  useEffect(() => {
    fetchCount({});
    fetchWines({}, 1);
  }, [fetchCount, fetchWines]);

  const handleApplyFilters = () => {
    const filters = buildFilters();
    console.log('Applying filters:', filters);
    setPage(1);
    fetchCount(filters);
    fetchWines(filters, 1);
    setDrawerOpen(false);
  };

  const handlePageChange = (_, value) => {
    const filters = buildFilters();
    setPage(value);
    fetchWines(filters, value);
  };

  const totalPages = Math.ceil(totalCount / 20);

  const FilterPanel = (
    <Box sx={{ width: { xs: 260, md: 300 }, p: 2 }}>
      <Typography variant="h6" gutterBottom>Filters</Typography>
      <Divider sx={{ mb:2 }} />

      <Typography variant="subtitle1">Wine Types</Typography>
      <FormGroup>
        {wineTypes.map(type => (
          <FormControlLabel
            key={type}
            control={
              <Checkbox
                checked={selectedTypes.includes(type)}
                onChange={e =>
                  setSelectedTypes(prev =>
                    e.target.checked
                      ? [...prev, type]
                      : prev.filter(t => t !== type)
                  )
                }
              />
            }
            label={type}
          />
        ))}
      </FormGroup>

      <Box sx={{ mt:3 }}>
        <Typography variant="subtitle1">Price Range</Typography>
        <Slider
          value={priceRange}
          onChange={(_, v) => setPriceRange(v)}
          valueLabelDisplay="auto"
          min={DEFAULT_PRICE_RANGE[0]}
          max={DEFAULT_PRICE_RANGE[1]}
        />
      </Box>

      <Box sx={{ mt:3 }}>
        <Typography variant="subtitle1">Min Rating</Typography>
        <Slider
          value={minRating}
          onChange={(_, v) => setMinRating(v)}
          valueLabelDisplay="auto"
          min={0}
          max={5}
          step={0.1}
        />
      </Box>

      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          freeSolo
          options={[]}
          value={selectedGrapes}
          onChange={(_, v) => setSelectedGrapes(v)}
          renderInput={params => <TextField {...params} label="Grapes" />}
        />
      </Box>

      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={regionsList}
          value={selectedRegions}
          onChange={(_, v) => setSelectedRegions(v)}
          renderInput={params => <TextField {...params} label="Regions" />}
        />
      </Box>

      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={countriesList}
          value={selectedCountries}
          onChange={(_, v) => setSelectedCountries(v)}
          renderInput={params => <TextField {...params} label="Countries" />}
        />
      </Box>

      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={stylesList}
          value={selectedStyles}
          onChange={(_, v) => setSelectedStyles(v)}
          renderInput={params => <TextField {...params} label="Styles" />}
        />
      </Box>

      <Box sx={{ mt:3 }}>
        <Autocomplete
          multiple
          options={pairingsList}
          value={selectedPairings}
          onChange={(_, v) => setSelectedPairings(v)}
          renderInput={params => <TextField {...params} label="Pairings" />}
        />
      </Box>

      <Button
        variant="contained"
        fullWidth
        sx={{ mt:4 }}
        onClick={handleApplyFilters}
      >
        Apply Filters
      </Button>
    </Box>
  );

  return (
    <Container sx={{ py:5 }}>
      <IconButton
        sx={{ display:{ md:'none' }, mb:2 }}
        onClick={()=>setDrawerOpen(true)}
      >
        <MenuIcon />
      </IconButton>

      <Box sx={{ display:{ xs:'block', md:'flex' }, gap:4 }}>
        <Box sx={{ display:{ xs:'none', md:'block' } }}>
          <Paper elevation={0} sx={{
            p:2,
            borderRadius:3,
            backdropFilter:'blur(10px)',
            backgroundColor:'rgba(255,255,255,0.05)',
            border:'1px solid rgba(255,255,255,0.15)'
          }}>
            {FilterPanel}
          </Paper>
        </Box>

        <Drawer
          open={drawerOpen}
          onClose={()=>setDrawerOpen(false)}
        >
          {FilterPanel}
        </Drawer>

        <Box sx={{ flex:1 }}>
          <Typography
            variant="h4"
            align="center"
            gutterBottom
            sx={{ textTransform:'uppercase', color:'text.primary' }}
          >
            Explore Wines
          </Typography>

          {loading ? (
            <Typography align="center">Loading…</Typography>
          ) : wines.length === 0 ? (
            <Typography align="center">No wines found.</Typography>
          ) : (
            <Box sx={{
              display:'grid',
              gridTemplateColumns:'repeat(auto-fit, minmax(220px,1fr))',
              gap:2
            }}>
              {wines.map(wine=>(
                <motion.div key={wine.id} whileHover={{scale:1.05}} transition={{duration:0.3}}>
                  <Card sx={{cursor:'pointer'}} onClick={()=>navigate(`/wine-details/${wine.id}`)}>
                    <CardMedia
                      component="img"
                      sx={{height:300, objectFit:'contain'}}
                      image={wine.image_url||'https://source.unsplash.com/random/400x300/?wine'}
                      alt={wine.wine_name}
                    />
                    <CardContent>
                      <Typography variant="h6" noWrap sx={{textTransform:'uppercase', color:'text.primary'}}>
                        {wine.wine_name}
                      </Typography>
                      <Typography variant="body2" noWrap sx={{textTransform:'uppercase', color:'text.secondary'}}>
                        ${wine.price} • {wine.rating} Stars
                      </Typography>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </Box>
          )}

          {!loading && totalPages > 1 && (
            <Box sx={{display:'flex', justifyContent:'center', mt:4}}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={handlePageChange}
                color="primary"
              />
            </Box>
          )}
        </Box>
      </Box>
    </Container>
  );
}
