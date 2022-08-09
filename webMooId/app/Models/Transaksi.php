<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Transaksi extends Model
{
    // Table Name
    protected $table = 'transaksi';
    // Primary Key
    public $primary_key = 'id';
    // Timestamps
    public $timestamps = false;

    protected $fillable = [
        'id_peternakan',
        'id_obat',
        'total_harga',
        'packaging',
        'tanggal_transaksi',
        'status',
    ];

    public function peternakan()
    {
        return $this->belongsTo('App\Models\Peternakan','id_peternakan');
    }

    public function obat()
    {
        return $this->belongsTo('App\Models\Obat','id_obat');
    }
}
