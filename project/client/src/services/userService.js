// client/src/services/userService.js
// Replace the mock implementations with real HTTP calls as your API becomes available

// Example with axios (uncomment if you install axios):
// import axios from 'axios';

export async function getUserProfile() {
    // Mock delay
    await new Promise((res) => setTimeout(res, 300));
  
    // Mocked user data
    return {
      id: 'user-123',
      name: 'John Doe',
      email: 'johndoe@example.com',
      avatarUrl: 'https://source.unsplash.com/random/100x100/?portrait',
      favorites: [
        { id: 'wine-1', name: 'Château Margaux', region: 'Bordeaux', vintage: 2015 },
        { id: 'wine-2', name: 'Opus One', region: 'Napa Valley', vintage: 2016 },
        { id: 'wine-3', name: 'Penfolds Grange', region: 'Australia', vintage: 2014 },
      ],
      purchases: [
        { id: 'order-1', wineName: 'Château Margaux', price: 299.99, date: '2024-02-15' },
        { id: 'order-2', wineName: 'Opus One', price: 249.5, date: '2024-03-10' },
        { id: 'order-3', wineName: 'Penfolds Grange', price: 349.0, date: '2024-01-05' },
      ],
      notes: [
        { id: 'note-1', date: '2024-02-16', text: 'Rich, full-bodied, with notes of blackberry and oak.' },
        { id: 'note-2', date: '2024-03-11', text: 'Smooth tannins, dark fruit, hint of vanilla.' },
      ],
    };
  
    // Real API example:
    // const res = await axios.get('/api/user/profile');
    // return res.data;
  }
  
  export async function getRecommendations(userId) {
    // Mock delay
    await new Promise((res) => setTimeout(res, 300));
  
    // Mocked recommendations
    return [
      {
        id: 'wine-4',
        name: 'Screaming Eagle',
        region: 'Napa Valley',
        vintage: 2018,
        imageUrl: 'https://source.unsplash.com/random/200x200/?wine',
      },
      {
        id: 'wine-5',
        name: 'Vega Sicilia Único',
        region: 'Ribera del Duero',
        vintage: 2012,
        imageUrl: 'https://source.unsplash.com/random/200x200/?wine-bottle',
      },
      {
        id: 'wine-6',
        name: 'Domaine de la Romanée-Conti',
        region: 'Burgundy',
        vintage: 2017,
        imageUrl: 'https://source.unsplash.com/random/200x200/?red-wine',
      },
    ];
  
    // Real API example:
    // const res = await axios.get(`/api/user/${userId}/recommendations`);
    // return res.data;
  }
  